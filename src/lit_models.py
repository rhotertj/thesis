import torch
import pytorch_lightning as pl
import pytorchvideo
import torchmetrics as tm
import wandb
from torchvision.transforms.functional import center_crop
import numpy as np
import seaborn as sns
import itertools

from utils import draw_trajectory, plot_confmat
from video_models import make_kinetics_mvit
from graph_models import GAT
from multimodal_models import MultiModalModel

from data.labels import LabelDecoder

class LitModel(pl.LightningModule):

    def __init__(
        self,
        learning_rate: float,
        label_mapping: LabelDecoder,
        momentum: float,
        weight_decay: float,
        max_epochs: int,
        loss_func : callable,
        model : torch.nn.Module
    ) -> None:
        super().__init__()
        num_classes = label_mapping.num_classes
        self.model = model

        wandb.watch(self.model)

        self.lr = learning_rate
        self.loss = eval(loss_func)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.class_names = label_mapping.class_names


        self.train_accuracy = tm.Accuracy(task="multiclass", threshold=0.5, num_classes=num_classes, multiclass=True, average=None)
        self.val_accuracy = tm.Accuracy(task="multiclass", threshold=0.5, num_classes=num_classes, multiclass=True, average=None)

        # cache, needed from lightning v2.0.0
        self.ground_truths = []
        self.predictions = []
        self.confidences = []
        self.val_loss = []
        self.train_loss = []

    def forward(self, input):
        if isinstance(self.model, pytorchvideo.models.vision_transformers.MultiscaleVisionTransformers):
            x = input["frames"]
            return self.model(x)
        elif isinstance(self.model, GAT):
            positions = input["positions"]
            return self.model(positions, positions.ndata["positions"])
        elif isinstance(self.model, MultiModalModel):
            return self.model(input)

    def training_step(self, batch, batch_idx):
        targets = batch["label"]
        offsets = batch["label_offset"]
        y = self.forward(batch)

        # loss = self.loss(y, targets)
        loss = self.loss(y, targets, offsets)
        self.log("train/batch_loss", loss)
        self.train_loss.append(loss.detach().cpu().item())

        if targets.ndim == 2:
            targets = targets.argmax(-1)

        self.train_accuracy.update(y, targets)
        self.log("train/batch_acc", torch.mean((targets == y.argmax(-1)).to(torch.float32)))
        return loss

    def on_train_epoch_start(self):
        self.train_accuracy.reset()
        self.train_loss.clear()

    def on_train_epoch_end(self):
        self.log("train/epoch_loss", np.mean(self.train_loss))
        acc_per_class = self.train_accuracy.compute()
        self.log("train/acc", torch.mean(acc_per_class))
        
        acc_dict = {f"train/acc_{n}" : acc_per_class[i] for i, n in enumerate(self.class_names)}
        self.log_dict(acc_dict)

    def on_validation_epoch_start(self) -> None:
        self.ground_truths.clear()
        self.predictions.clear()
        self.confidences.clear()
        self.val_loss.clear()
        
        self.val_accuracy.reset()

    def validation_step(self, batch, batch_idx):
        targets = batch["label"]
        offsets = batch["label_offset"]
        y = self.forward(batch)
        # loss = self.loss(y, targets)
        loss = self.loss(y, targets, offsets)
        self.log("val/batch_loss", loss.mean())

        self.val_accuracy.update(y, targets)
        preds = y.detach().cpu()

        # save prediction and ground truth for validation metrics
        for b in range(len(targets)):
            self.predictions.append(preds[b].argmax().item())
            self.confidences.append(preds[b])
            self.ground_truths.append(targets[b].detach().cpu().item())
        self.val_loss.append(loss.detach().cpu().item())


    def on_validation_epoch_end(self) -> None:
        acc_per_class = self.val_accuracy.compute()
        self.log("val/acc", torch.mean(acc_per_class))
        acc_dict = {f"val/acc_{n}" : acc_per_class[i] for i, n in enumerate(self.class_names)}
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
    return torch.nn.functional.cross_entropy(batch_y, batch_gt, label_smoothing=0.1)

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
    batch_losses = torch.nn.functional.cross_entropy(batch_y, batch_gt, reduction='none', label_smoothing=0.1)
    for i, o in enumerate(batch_label_offsets):
        if o > 0:
            batch_losses[i] *= 1/o
    return batch_losses.mean()