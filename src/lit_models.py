import torch
import pytorch_lightning as pl
import pytorchvideo
import torchmetrics as tm
import wandb
from torchvision.transforms.functional import center_crop
import numpy as np
import seaborn as sns
import itertools
import pickle as pkl

from video_models import make_kinetics_mvit
from graph_models import GAT, PositionTransformer, GIN, GCN
from multimodal_models import MultiModalModel

from metrics import average_mAP, postprocess_predictions
from data.labels import LabelDecoder


class LitModel(pl.LightningModule):

    def __init__(
        self,
        label_mapping: LabelDecoder,
        scheduler: callable,
        optimizer: callable,
        loss_func: callable,
        model: torch.nn.Module
    ) -> None:
        super().__init__()
        num_classes = label_mapping.num_classes
        self.model = model

        #wandb.watch(self.model)

        self.loss_func = loss_func
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.class_names = label_mapping.class_names

        self.train_accuracy = tm.Accuracy(task="multiclass", threshold=0.5, num_classes=num_classes, average=None)
        self.weighted_train_accuracy = tm.Accuracy(task="multiclass", threshold=0.5, num_classes=num_classes)

        self.val_accuracy = tm.Accuracy(task="multiclass", threshold=0.5, num_classes=num_classes, average=None)
        self.weighted_val_accuracy = tm.Accuracy(task="multiclass", threshold=0.5, num_classes=num_classes)

        # cache, needed from lightning v2.0.0
        self.ground_truths = []
        self.predictions = []
        self.confidences = []
        self.pred_frame_idx = []
        self.pred_match_number = []

        self.ground_truth_anchors = []
        self.ground_truth_anchors_label = []
        self.ground_truth_anchors_match_number = []

        self.val_loss = []
        self.train_loss = []

    def forward(self, input):
        if isinstance(self.model, pytorchvideo.models.vision_transformers.MultiscaleVisionTransformers):
            x = input["frames"]
            return self.model(x)
        elif isinstance(self.model, (GAT, GIN)):
            positions = input["positions"]
            return self.model(positions, positions.ndata["positions"])
        elif isinstance(self.model, GCN):
            positions = input["positions"]
            return self.model(positions, positions.ndata["positions"], positions.edata["w"])
        elif isinstance(self.model, PositionTransformer):
            positions = input["positions"]
            return self.model(positions)
        elif isinstance(self.model, MultiModalModel):
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
        self.ground_truths.clear()
        self.predictions.clear()
        self.confidences.clear()
        self.pred_frame_idx.clear()
        self.pred_match_number.clear()
        self.ground_truth_anchors.clear()
        self.ground_truth_anchors_label.clear()
        self.ground_truth_anchors_match_number.clear()
        self.val_loss.clear()
        self.weighted_val_accuracy.reset()
        self.val_accuracy.reset()

    def validation_step(self, batch, batch_idx):
        targets = batch["label"]
        offsets = batch["label_offset"]
        match_numbers = batch["match_number"]
        frame_indices = batch["frame_idx"]
        y = self.forward(batch)

        if isinstance(y, tuple):
            y, y_reg = y
            loss = self.loss_func(y, y_reg, targets, offsets)
        else:
            loss = self.loss_func(y, targets, offsets)
        self.log("val/batch_loss", loss.mean())

        self.val_accuracy.update(y, targets)
        self.weighted_val_accuracy.update(y, targets)
        preds = y.detach().cpu()

        # save for validation metrics
        for b in range(len(targets)):
            # all predictions for PR curve and confmat, frame_idx for post_processing
            self.predictions.append(preds[b].argmax().item())
            self.confidences.append(preds[b])
            self.ground_truths.append(targets[b].detach().cpu().item())
            self.pred_frame_idx.append(frame_indices[b].cpu().item())
            self.pred_match_number.append(match_numbers[b].cpu().item())
            # gather ground truth for instances with annotated actions at pos 0.
            if offsets[b] == 0: # this includes background predictions, they will be discarded later 
                self.ground_truth_anchors.append(frame_indices[b].cpu().item())
                self.ground_truth_anchors_match_number.append(match_numbers[b].cpu().item())
                self.ground_truth_anchors_label.append(targets[b].detach().cpu().item())
        self.val_loss.append(loss.detach().cpu().item())

    def on_validation_epoch_end(self) -> None:
        acc_per_class = self.val_accuracy.compute()
        acc = self.weighted_val_accuracy.compute()
        self.log("val/acc", acc)

        acc_dict = {f"val/acc_{n}": acc_per_class[i] for i, n in enumerate(self.class_names)}
        self.log_dict(acc_dict)

        self.log("val/loss", np.mean(self.val_loss))

        cm = wandb.plot.confusion_matrix(
            y_true=self.ground_truths,
            preds=self.predictions,
            class_names=self.class_names,
            title="Confusion Matrix",
        )

        wandb.log({"val/conf_mat": cm})

        confidences = torch.stack(self.confidences)
        pr = wandb.plot.pr_curve(
            self.ground_truths,
            confidences,
            labels=self.class_names,
            title="Precision vs. Recall per class",
        )

        wandb.log({"val/prec_rec": pr})

        # offset ground truth anchor positions to avoid collisions across matches
        max_frame_magnitude = len(str(max(self.ground_truth_anchors)))
        frame_offset = 10**(max_frame_magnitude + 1)

        gt_match_number = np.array(self.ground_truth_anchors_match_number)
        gt_anchor = np.array(self.ground_truth_anchors)
        gt_anchor = gt_anchor + frame_offset * gt_match_number
        gt_label = np.array(self.ground_truth_anchors_label)

        # offset frames for predictions as well
        pred_frame_idx = np.array(self.pred_frame_idx)
        match_number = np.array(self.pred_match_number)
        confidences = confidences.numpy()

        max_frame_magnitude = len(str(max(self.pred_frame_idx)))
        frame_offset = 10**(max_frame_magnitude + 1)

        altered_frames = pred_frame_idx + frame_offset * match_number
        correct_order = np.argsort(altered_frames)
        print(len(altered_frames), len(confidences), len(correct_order))
        altered_frames = altered_frames[correct_order]
        confidences = confidences[correct_order]

        pred_anchors, pred_confidences = postprocess_predictions(confidences, altered_frames)

        for var, name in zip([pred_confidences, pred_anchors, gt_label, gt_anchor], ["litmodel_confs", "litmodel_pred_anchors", "litmodel_gt_label", "litmodel_gt_anhors"]):
            with open(name+".pkl", "wb+") as f:
                pkl.dump(var, f)

        fps = 29.97
        tolerances = [fps * i for i in range(1,6)]
        map_per_tolerance = average_mAP(pred_confidences, pred_anchors, gt_label, gt_anchor, tolerances=tolerances)

        data = [[sec, m] for (sec, m) in zip(range(1,6), map_per_tolerance)]
        table = wandb.Table(data=data, columns = ["seconds", "mAP"])
        wandb.log(
            {"val/average-mAP" : wandb.plot.line(table, "seconds", "mAP", title=f"mAP at different tolerances (a-mAP={np.mean(map_per_tolerance):.2})")}
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
