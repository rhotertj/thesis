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
        max_epochs: int
    ) -> None:
        super().__init__()
        num_classes = label_mapping.num_classes
        self.model = make_kinetics_mvit(pretrained_path, num_classes)

        # if self.log is not None:
        #     wandb.watch(self.model)

        self.lr = learning_rate
        self.loss = torch.nn.CrossEntropyLoss()
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        self.train_accuracy = tm.Accuracy(task="multiclass", threshold=0.5, num_classes=num_classes, multiclass=True)

        self.val_accuracy = tm.Accuracy(task="multiclass", threshold=0.5, num_classes=num_classes, multiclass=True)

        # only use for validation!
        self.ground_truths = []
        self.predictions = []
        self.confidences = []
        self.class_names = label_mapping.class_names

    def forward(self, input):
        x = input["frames"]
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # print(batch)
        x = batch["frames"]
        targets = batch["label"]
        y = self.model(x)

        loss = self.loss(y, targets)
        self.log("train/batch_loss", loss)

        batch_acc = self.train_accuracy(y, targets)
        self.log("train/batch_acc", batch_acc)
        return loss

    def training_epoch_end(self, outputs):
        self.log("train/epoch_loss", np.mean([item["loss"].item() for item in outputs]))
        self.log("train/epoch_acc", self.train_accuracy.compute())
        self.train_accuracy.reset()

    def validation_step(self, batch, batch_idx):
        x = batch["frames"]
        targets = batch["label"]
        y = self.model(x)

        loss = self.loss(y, targets)
        self.log("val/batch_loss", loss)

        self.val_accuracy.update(y, targets)
        soft_y = y.detach().cpu().softmax(dim=-1)
        # save predicition and ground truth for validation metrics
        for b in range(x.shape[0]):
            self.predictions.append(soft_y[b].argmax().item())
            self.confidences.append(soft_y[b])
            self.ground_truths.append(targets[b].detach().cpu().item())

        return {"loss": loss.item()}

    def validation_epoch_end(self, outputs) -> None:
        self.log("val/acc", self.val_accuracy.compute())
        self.log("val/loss", np.mean([output["loss"] for output in outputs]))

        cm = wandb.plot.confusion_matrix(
            y_true=self.ground_truths,
            preds=self.predictions,
            class_names=self.class_names,
            title="Confusion Matrix",
        )

        wandb.log({"val/conf_mat": cm})

        self.confidences = torch.stack(self.confidences)
        pr = wandb.plot.pr_curve(
            self.ground_truths,
            self.confidences,
            labels=self.class_names,
            title="Precision vs. Recall per class",
        )

        wandb.log({"val/prec_rec": pr})

        self.ground_truths = []
        self.predictions = []
        self.confidences = []
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
        epsilon : float,
        dim_in: int,
        dim_h: int,
        heads : int,
        input_embedding: bool,
        readout: str,
        learning_rate: float,
        label_mapping: LabelDecoder,
        momentum: float,
        weight_decay: float,
        max_epochs: int
    ) -> None:
        super().__init__()
        num_classes = label_mapping.num_classes
        self.epsilon = epsilon

        self.model = GAT(
            dim_h=dim_h,
            dim_in=dim_in,
            num_classes=num_classes,
            readout=readout,
            heads=heads,
            input_embedding=input_embedding
        )

        self.lr = learning_rate
        self.loss = torch.nn.CrossEntropyLoss()
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        self.train_accuracy = tm.Accuracy(task="multiclass", threshold=0.5, num_classes=num_classes, multiclass=True)

        self.val_accuracy = tm.Accuracy(task="multiclass", threshold=0.5, num_classes=num_classes, multiclass=True)

        # only use for validation!
        self.ground_truths = []
        self.predictions = []
        self.confidences = []
        self.class_names = label_mapping.class_names

    def forward(self, input):
        positions = input["positions"].to(torch.float32)
        graphs = []
        for b in range(positions.shape[0]):
            g = create_graph(positions[b], 7)
            graphs.append(g)
        graph_batch = dgl.batch(graphs)
        res = self.model(graph_batch, graph_batch.ndata["positions"])
        return res

    def training_step(self, batch, batch_idx):
        targets = batch["label"]
        y = self.forward(batch)
        loss = self.loss(y, targets)
        self.log("train/batch_loss", loss)

        batch_acc = self.train_accuracy(y, targets)
        self.log("train/batch_acc", batch_acc)
        return loss

    def training_epoch_end(self, outputs):
        self.log("train/epoch_loss", np.mean([item["loss"].item() for item in outputs]))
        self.log("train/epoch_acc", self.train_accuracy.compute())
        self.train_accuracy.reset()

    def validation_step(self, batch, batch_idx):
        targets = batch["label"]
        y = self.forward(batch)
        loss = self.loss(y, targets)
        self.log("val/batch_loss", loss)

        self.val_accuracy.update(y, targets)
        soft_y = y.detach().cpu().softmax(dim=-1)
        # save predicition and ground truth for validation metrics
        for b in range(batch["positions"].shape[0]):
            self.predictions.append(soft_y[b].argmax().item())
            self.confidences.append(soft_y[b])
            self.ground_truths.append(targets[b].detach().cpu().item())

        return {"loss": loss.item()}

    def validation_epoch_end(self, outputs) -> None:
        self.log("val/acc", self.val_accuracy.compute())
        self.log("val/loss", np.mean([output["loss"] for output in outputs]))

        cm = wandb.plot.confusion_matrix(
            y_true=self.ground_truths,
            preds=self.predictions,
            class_names=self.class_names,
            title="Confusion Matrix",
        )

        wandb.log({"val/conf_mat": cm})

        self.confidences = torch.stack(self.confidences)
        pr = wandb.plot.pr_curve(
            self.ground_truths,
            self.confidences,
            labels=self.class_names,
            title="Precision vs. Recall per class",
        )

        wandb.log({"val/prec_rec": pr})

        self.ground_truths = []
        self.predictions = []
        self.confidences = []
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