from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import pytorch_lightning as pl
import pytorchvideo
import torchmetrics as tm
import wandb
import numpy as np
import pickle as pkl
from pathlib import Path

from video_models import make_kinetics_mvit
from graph_models import GAT, PositionTransformer, GIN, GCN
from multimodal_models import MultiModalModel, NetVLADModel

from metrics import average_mAP, nms_peaks, reorder_predictions
from data import LabelDecoder


class LitModel(pl.LightningModule):

    def __init__(
        self,
        label_mapping: LabelDecoder,
        scheduler: callable,
        optimizer: callable,
        loss_func: callable,
        model: torch.nn.Module,
        experiment_dir : Path
    ) -> None:
        super().__init__()
        num_classes = label_mapping.num_classes
        self.model = model

        #wandb.watch(self.model)
        self.experiment_dir = experiment_dir

        self.loss_func = loss_func
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.class_names = label_mapping.class_names

        self.train_accuracy = tm.Accuracy(task="multiclass", threshold=0.5, num_classes=num_classes, average=None)
        self.weighted_train_accuracy = tm.Accuracy(task="multiclass", threshold=0.5, num_classes=num_classes)

        self.val_accuracy = tm.Accuracy(task="multiclass", threshold=0.5, num_classes=num_classes, average=None)
        self.weighted_val_accuracy = tm.Accuracy(task="multiclass", threshold=0.5, num_classes=num_classes)

        self.test_accuracy = tm.Accuracy(task="multiclass", threshold=0.5, num_classes=num_classes, average=None)
        self.weighted_test_accuracy = tm.Accuracy(task="multiclass", threshold=0.5, num_classes=num_classes)

        # cache, needed from lightning v2.0.0
        self.cache = Cache(
            ground_truths = [],
            predictions = [],
            confidences = [],
            frame_idx = [],
            match_numbers = [],
            action_idx = [],
            label_offsets=[],
            loss=[]
        )
        self.val_loss = []
        self.train_loss = []
        self.test_loss = []

    def forward(self, input):
        if isinstance(self.model, pytorchvideo.models.vision_transformers.MultiscaleVisionTransformers):
            x = input["frames"]
            return self.model(x)
        elif isinstance(self.model, (GAT, GIN)):
            positions = input["positions"]
            return self.model(positions, positions.ndata["positions"])
        elif isinstance(self.model, GCN):
            positions = input["positions"]
            return self.model(positions, positions.ndata["positions"], positions.edata.get("w", None))
        elif isinstance(self.model, PositionTransformer):
            positions = input["positions"]
            return self.model(positions)
        elif isinstance(self.model, (MultiModalModel, NetVLADModel)):
            return self.model(input)
        else:
           print(f"Unknown model type {type(self.model)}")
           return self.model(input)

    def training_step(self, batch, batch_idx):
        targets = batch["label"]
        offsets = batch["label_offset"]
        y = self.forward(batch)

        if isinstance(y, tuple):
            y, y_reg = y
            loss = self.loss_func(y, y_reg, targets, offsets)
        else:
            loss = self.loss_func(y, targets, offsets)

        self.log("train/batch_loss", loss)
        self.train_loss.append(loss.detach().cpu().item())

        if targets.ndim == 2:
            targets = targets.argmax(-1)

        self.train_accuracy.update(y, targets)
        self.weighted_train_accuracy.update(y, targets)
        self.log("train/batch_acc", torch.mean((targets == y.argmax(-1)).to(torch.float32)))
        return loss

    def on_train_epoch_start(self):
        self.train_accuracy.reset()
        self.weighted_train_accuracy.reset()
        self.train_loss.clear()

    def on_train_epoch_end(self):
        self.log("train/epoch_loss", np.mean(self.train_loss))
        acc_per_class = self.train_accuracy.compute()
        acc = self.weighted_train_accuracy.compute()
        self.log("train/acc", acc)

        acc_dict = {f"train/acc_{n}": acc_per_class[i] for i, n in enumerate(self.class_names)}
        self.log_dict(acc_dict)

    def on_validation_epoch_start(self) -> None:
        self.cache.clear()

        self.val_loss.clear()
        self.weighted_val_accuracy.reset()
        self.val_accuracy.reset()

    def validation_step(self, batch, batch_idx):
        targets = batch["label"]
        offsets = batch["label_offset"]
        y = self.forward(batch)

        if isinstance(y, tuple):
            y, y_reg = y
            loss = self.loss_func(y, y_reg, targets, offsets)
        else:
            loss = self.loss_func(y, targets, offsets)
        log_loss = torch.nn.functional.cross_entropy(y, targets, label_smoothing=0.1, reduction='none')
        self.log("val/batch_loss", loss.mean())

        self.val_accuracy.update(y, targets)
        self.weighted_val_accuracy.update(y, targets)
        preds = y.detach().cpu()
        # save for validation metrics
        self.cache.update(
            predictions=preds.argmax(-1),
            confidences=preds,
            ground_truths=targets,
            frame_idx=batch["frame_idx"],
            match_numbers=batch["match_number"],
            action_idx=batch["label_idx"],
            label_offsets=offsets,
            loss=log_loss
        )

        self.val_loss.append(loss.detach().cpu().item())

    def on_validation_epoch_end(self) -> None:
        acc_per_class = self.val_accuracy.compute()
        acc = self.weighted_val_accuracy.compute()
        self.log("val/acc", acc)

        acc_dict = {f"val/acc_{n}": acc_per_class[i] for i, n in enumerate(self.class_names)}
        self.log_dict(acc_dict)

        self.log("val/loss", np.mean(self.val_loss))

        self.cache.save(self.experiment_dir / "val_results.pkl")

        cm = wandb.plot.confusion_matrix(
            y_true=self.cache.get("ground_truths"),
            preds=self.cache.get("predictions"),
            class_names=self.class_names,
            title="Confusion Matrix",
        )

        wandb.log({"val/conf_mat": cm})

        confidences = torch.stack(self.cache.get("confidences", as_numpy=False))
        pr = wandb.plot.pr_curve(
            self.cache.get("ground_truths"),
            confidences,
            labels=self.class_names,
            title="Precision vs. Recall per class",
        )

        wandb.log({"val/prec_rec": pr})

        confidences, confs_frames, anchors, anchor_labels = reorder_predictions(self.cache)
        pred_anchors, pred_confidences = nms_peaks(confidences, confs_frames)

        fps = 29.97
        tolerances = [fps * i for i in range(1,6)]
        map_per_tolerance = average_mAP(pred_confidences, pred_anchors, anchor_labels, anchors, tolerances=tolerances)

        data = [[sec, m] for (sec, m) in zip(range(1,6), map_per_tolerance)]
        table = wandb.Table(data=data, columns = ["seconds", "mAP"])
        wandb.log(
            {"val/average-mAP" : wandb.plot.line(table, "seconds", "mAP", title=f"mAP at different tolerances (a-mAP={np.mean(map_per_tolerance):.2})")}
        )

    def on_test_epoch_start(self) -> None:
        self.cache.clear()

        self.test_loss.clear()
        self.weighted_test_accuracy.reset()
        self.test_accuracy.reset()

    def test_step(self, batch, batch_index):
        targets = batch["label"]
        offsets = batch["label_offset"]
        y = self.forward(batch)

        if isinstance(y, tuple):
            y, y_reg = y
            loss = self.loss_func(y, y_reg, targets, offsets)
        else:
            loss = self.loss_func(y, targets, offsets)
        log_loss = torch.nn.functional.cross_entropy(y, targets, label_smoothing=0.1, reduction='none')
        self.log("test/batch_loss", loss.mean())

        self.test_accuracy.update(y, targets)
        self.weighted_test_accuracy.update(y, targets)
        preds = y.detach().cpu()
        # save for test metrics
        self.cache.update(
            predictions=preds.argmax(-1),
            confidences=preds,
            ground_truths=targets,
            frame_idx=batch["frame_idx"],
            match_numbers=batch["match_number"],
            action_idx=batch["label_idx"],
            label_offsets=offsets,
            loss=log_loss
        )

        self.test_loss.append(loss.detach().cpu().item())

    def on_test_epoch_end(self) -> None:
        acc_per_class = self.test_accuracy.compute()
        acc = self.weighted_test_accuracy.compute()
        self.log("test/acc", acc)

        acc_dict = {f"test/acc_{n}": acc_per_class[i] for i, n in enumerate(self.class_names)}
        self.log_dict(acc_dict)

        self.log("test/loss", np.mean(self.test_loss))

        self.cache.save(self.experiment_dir / "test_results.pkl")

        cm = wandb.plot.confusion_matrix(
            y_true=self.cache.get("ground_truths"),
            preds=self.cache.get("predictions"),
            class_names=self.class_names,
            title="Confusion Matrix",
        )

        wandb.log({"test/conf_mat": cm})

        confidences = torch.stack(self.cache.get("confidences", as_numpy=False))
        pr = wandb.plot.pr_curve(
            self.cache.get("ground_truths"),
            confidences,
            labels=self.class_names,
            title="Precision vs. Recall per class",
        )

        wandb.log({"test/prec_rec": pr})

        confidences, confs_frames, anchors, anchor_labels = reorder_predictions(self.cache)
        pred_anchors, pred_confidences = nms_peaks(confidences, confs_frames)

        fps = 29.97
        tolerances = [fps * i for i in range(1,6)]
        map_per_tolerance = average_mAP(pred_confidences, pred_anchors, anchor_labels, anchors, tolerances=tolerances)

        data = [[sec, m] for (sec, m) in zip(range(1,6), map_per_tolerance)]
        table = wandb.Table(data=data, columns = ["seconds", "mAP"])
        wandb.log(
            {"test/average-mAP" : wandb.plot.line(table, "seconds", "mAP", title=f"mAP at different tolerances (a-mAP={np.mean(map_per_tolerance):.2})")}
        )


    def configure_optimizers(self):
        if self.scheduler is None:
            return self.optimizer
        return [self.optimizer], [self.scheduler]


def unweighted_cross_entropy(batch_y: torch.Tensor, batch_gt: torch.Tensor, batch_label_offsets: torch.Tensor = None):
    """Computes the cross entropy loss across a batch.

    Args:
        batch_y (torch.Tensor): Predictions.
        batch_gt (torch.Tensor): Targets.
        batch_label_offsets (torch.Tensor): Label offsets.

    Returns:
        torch.Tensor: Loss.
    """
    return torch.nn.functional.cross_entropy(batch_y, batch_gt, label_smoothing=0.1)


def weighted_cross_entropy(batch_y: torch.Tensor, batch_gt: torch.Tensor, batch_label_offsets: torch.Tensor):
    """Computes the cross entropy loss between prediction and target.
    Instances that portray an event in the second half of the sequence have reduced losses. 

    Args:
        batch_y (torch.Tensor): Predictions.
        batch_gt (torch.Tensor): Targets.
        batch_label_offsets (torch.Tensor): Label offsets.

    Returns:
        (torch.Tensor): Loss.
    """
    batch_losses = torch.nn.functional.cross_entropy(batch_y, batch_gt, reduction='none', label_smoothing=0.1)
    for i, o in enumerate(batch_label_offsets):
        if o > 0:
            batch_losses[i] *= 1 / o
    return batch_losses.mean()

def twin_head_loss(batch_cls: torch.Tensor, batch_reg: torch.Tensor, batch_gt: torch.Tensor, batch_label_offsets: torch.Tensor):
    cls_loss = torch.nn.functional.cross_entropy(batch_cls, batch_gt, label_smoothing=0.1)
    reg_loss = torch.nn.functional.huber_loss(batch_reg.squeeze(1), batch_label_offsets.float())
    sigma = 0.8
    return cls_loss + (sigma * reg_loss)


class Cache:

    def __init__(self, **kwargs):
        """A cache for concatenating [B,1] shaped arrays in one list.
        """        
        self.data = {}
        for k, v in kwargs.items():
            self.data[k] = v

    def update(self, batched=True, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu()

                if batched:
                    for b in range(v.shape[0]):
                        if v[b].ndim == 0 or len(v[b]) == 1:
                            self.data[k].append(v[b].item())
                        else:
                            self.data[k].append(v[b])
            else:
                self.data[k].append(v)

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self.data[k] = v

    def get(self, key, as_numpy=True):
        if as_numpy:
            return np.array(self.data[key])
        return self.data[key]

    def save(self, path : Path):
        with open(path, "wb+") as f:
            pkl.dump(self.data, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            loaded = pkl.load(f)
        for k, v in loaded.items():
            if k in self.data:
                print("Overwriting", k, "from loaded cache.")
            self.data[k] = v      

    def clear(self):
        for k in self.data.keys():
            self.data[k] = []  

    def __repr__(self):
        return f"{self.data=}"

if __name__ == "__main__":
    cache = Cache(listig=[], omnom=["Banane"])
    cache.update(omnom="Kaffee")
    cache.update(listig=torch.rand(8,1))
    cache.save("./cache.pkl")
    print(cache)
