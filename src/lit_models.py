import torch
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
import torchmetrics as tm
import wandb
from torchvision.transforms.functional import center_crop
import numpy as np
import seaborn as sns
import itertools

from utils import draw_trajectory, plot_confmat
from video_models import make_kinetics_mvit
from graph_models import GAT
import dgl
from data.labels import LabelDecoder
from data.data_utils import create_graph

class LitMViT(pl.LightningModule):

    def __init__(
        self,
        pretrained_path: str,
        learning_rate: float,
        label_mapping: LabelDecoder,
        momentum: float,
        weight_decay: float,
        max_epochs: int,
        loss_func : callable
    ) -> None:
        super().__init__()
        num_classes = label_mapping.num_classes
        self.model = make_kinetics_mvit(pretrained_path, num_classes)

        self.lr = learning_rate
        self.loss = eval(loss_func)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        self.train_accuracy = tm.Accuracy(task="multiclass", threshold=0.5, num_classes=num_classes, multiclass=True, average=None)
        self.val_accuracy = tm.Accuracy(task="multiclass", threshold=0.5, num_classes=num_classes, multiclass=True, average=None)
        
        self.class_names = label_mapping.class_names

    def forward(self, input):
        x = input["frames"]
        return self.model(x)

    def training_step(self, batch, batch_idx):
        targets = batch["label"]
        offsets = batch["label_offset"]
        y = self.forward(batch)

        loss = self.loss(y, targets, offsets)
        self.log("train/batch_loss", loss.mean())

        if targets.ndim == 2:
            targets = targets.argmax(-1)

        self.train_accuracy.update(y, targets)
        self.log("train/batch_acc", torch.mean((targets == y.argmax(-1)).to(torch.float32)))
        return loss

    def training_epoch_end(self, outputs):
        self.log("train/epoch_loss", np.mean([item["loss"].item() for item in outputs]))
        acc_per_class = self.train_accuracy.compute()
        self.log("train/acc", torch.mean(acc_per_class))
        
        acc_dict = {n : acc_per_class[i] for i, n in enumerate(self.class_names)}
        self.log("train/acc_classes", acc_dict)

        self.train_accuracy.reset()

    def validation_step(self, batch, batch_idx):
        targets = batch["label"]
        offsets = batch["label_offset"]
        y = self.forward(batch)
        loss = self.loss(y, targets, offsets)

        self.log("val/batch_loss", loss.mean())

        self.val_accuracy.update(y, targets)
        preds = y.detach().cpu()

        # save prediction and ground truth for validation metrics
        ground_truths = []
        predictions = []
        confidences = []
        for b in range(len(targets)):
            predictions.append(preds[b].argmax().item())
            confidences.append(preds[b])
            ground_truths.append(targets[b].detach().cpu().item())

        return {"loss": loss, "predictions" : predictions, "confidences" : confidences, "ground_truths" : ground_truths}

    def validation_epoch_end(self, outputs) -> None:
        losses = []
        confidences = []
        predictions = []
        targets = []
        for output in outputs:
            losses.append(output["loss"].item())
            confidences.extend(output["confidences"])
            predictions.extend(output["predictions"])
            targets.extend(output["ground_truths"])

        acc_per_class = self.val_accuracy.compute()
        self.log("val/acc", torch.mean(acc_per_class))
        acc_dict = {n : acc_per_class[i] for i, n in enumerate(self.class_names)}
        self.log("val/acc_per_class", acc_dict)
        self.log("val/loss", np.mean(losses))

        cm = wandb.plot.confusion_matrix(
            y_true=targets,
            preds=predictions,
            class_names=self.class_names,
            title="Confusion Matrix",
        )

        wandb.log({"val/conf_mat": cm})

        confidences = torch.stack(confidences)
        pr = wandb.plot.pr_curve(
            targets,
            confidences,
            labels=self.class_names,
            title="Precision vs. Recall per class",
        )

        wandb.log({"val/prec_rec": pr})

        self.val_accuracy.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=(self.lr or self.learning_rate),
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            self.max_epochs,
            last_epoch=-1,
        )
        return [optimizer], [scheduler]

class LitGAT(pl.LightningModule):

    def __init__(
        self,
        dim_in: int,
        dim_h: int,
        heads : int,
        input_embedding: bool,
        readout: str,
        learning_rate: float,
        label_mapping: LabelDecoder,
        momentum: float,
        weight_decay: float,
        max_epochs: int,
        loss_func: callable
    ) -> None:
        super().__init__()
        num_classes = label_mapping.num_classes

        self.model = GAT(
            dim_h=dim_h,
            dim_in=dim_in,
            num_classes=num_classes,
            readout=readout,
            heads=heads,
            input_embedding=input_embedding
        )

        self.lr = learning_rate
        self.loss = eval(loss_func)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        self.train_accuracy = tm.Accuracy(task="multiclass", threshold=0.5, num_classes=num_classes, multiclass=True, average=None)
        self.val_accuracy = tm.Accuracy(task="multiclass", threshold=0.5, num_classes=num_classes, multiclass=True, average=None)
        
        self.class_names = label_mapping.class_names

    def forward(self, input):
        positions = input["positions"]
        res = self.model(positions, positions.ndata["positions"])
        return res

    def training_step(self, batch, batch_idx):
        targets = batch["label"]
        offsets = batch["label_offset"]
        y = self.forward(batch)
        loss = self.loss(y, targets, offsets)
        self.log("train/batch_loss", loss.mean())

        self.train_accuracy.update(y, targets)
        self.log("train/batch_acc", torch.mean((targets == y.argmax(-1)).to(torch.float32)))
        return loss

    def training_epoch_end(self, outputs):
        self.log("train/epoch_loss", np.mean([item["loss"].item() for item in outputs]))
        acc_per_class = self.train_accuracy.compute()
        self.log("train/acc", torch.mean(acc_per_class))
        
        acc_dict = {n : acc_per_class[i] for i, n in enumerate(self.class_names)}
        self.log("train/acc_classes", acc_dict)

        self.train_accuracy.reset()

    def validation_step(self, batch, batch_idx):
        targets = batch["label"]
        offsets = batch["label_offset"]
        y = self.forward(batch)
        loss = self.loss(y, targets, offsets)

        self.log("val/batch_loss", loss.mean())

        self.val_accuracy.update(y, targets)
        preds = y.detach().cpu()

        # save prediction and ground truth for validation metrics
        ground_truths = []
        predictions = []
        confidences = []
        for b in range(len(targets)):
            predictions.append(preds[b].argmax().item())
            confidences.append(preds[b])
            ground_truths.append(targets[b].detach().cpu().item())

        return {"loss": loss, "predictions" : predictions, "confidences" : confidences, "ground_truths" : ground_truths}

    def validation_epoch_end(self, outputs) -> None:
        losses = []
        confidences = []
        predictions = []
        targets = []
        for output in outputs:
            losses.append(output["loss"].item())
            confidences.extend(output["confidences"])
            predictions.extend(output["predictions"])
            targets.extend(output["ground_truths"])

        acc_per_class = self.val_accuracy.compute()
        self.log("val/acc", torch.mean(acc_per_class))
        acc_dict = {n : acc_per_class[i] for i, n in enumerate(self.class_names)}
        self.log("val/acc_per_class", acc_dict)
        self.log("val/loss", np.mean(losses))

        cm = wandb.plot.confusion_matrix(
            y_true=targets,
            preds=predictions,
            class_names=self.class_names,
            title="Confusion Matrix",
        )

        wandb.log({"val/conf_mat": cm})

        confidences = torch.stack(confidences)
        pr = wandb.plot.pr_curve(
            targets,
            confidences,
            labels=self.class_names,
            title="Precision vs. Recall per class",
        )

        wandb.log({"val/prec_rec": pr})

        self.val_accuracy.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=(self.lr or self.learning_rate),
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            self.max_epochs,
            last_epoch=-1,
        )
        return [optimizer], [scheduler]

def unweighted_cross_entropy(batch_y : torch.Tensor, batch_gt : torch.Tensor, batch_label_offsets : torch.Tensor = None):
    """Computes the cross entropy loss across a batch.

    Args:
        batch_y (torch.Tensor): Predictions.
        batch_gt (torch.Tensor): Targets.
        batch_label_offsets (torch.Tensor): Label offsets.

    Returns:
        torch.Tensor: Loss.
    """    
    return torch.nn.functional.cross_entropy(batch_y, batch_gt)

def weighted_cross_entropy(batch_y : torch.Tensor, batch_gt : torch.Tensor, batch_label_offsets : torch.Tensor):
    """Computes the cross entropy loss between prediction and target.
    Instances that portray an event in the second half of the sequence have reduced losses. 

    Args:
        batch_y (torch.Tensor): Predictions.
        batch_gt (torch.Tensor): Targets.
        batch_label_offsets (torch.Tensor): Label offsets.

    Returns:
        (torch.Tensor): Loss.
    """
    batch_losses = torch.nn.functional.cross_entropy(batch_y, batch_gt, reduction='none')
    for i, o in enumerate(batch_label_offsets):
        if o > 0:
            batch_losses[i] *= 1/o
    return batch_losses.mean()