import torch
from torch import nn

import pytorch_lightning as pl
import torchmetrics
from modti.utils import get_activation
from modti.models.layers import AVAILABLE_LAYERS


class Monolithic(pl.LightningModule):
    def __init__(
        self,
        input_dim=2048,
        task_dim=100,
        latent_dim=1024,
        activation='ReLU',
        classify=True,
        last_layer_type="Cosine",
        lr=1e-4,
        **last_layer_params
    ):
        super().__init__()

        self.input_dim = input_dim
        self.task_dim = task_dim
        self.latent_dim = latent_dim
        self.activation = get_activation(activation)
        self.pred_layer = AVAILABLE_LAYERS[last_layer_type](**last_layer_params)

        self.classify = classify
        self.lr = lr

        self.input_projector = nn.Sequential(nn.Linear(self.input_dim, self.latent_dim), self.activation)
        self.task_projector = nn.Sequential(nn.Linear(self.task_dim, self.latent_dim), self.activation)

        if self.classify:
            self.val_accuracy = torchmetrics.Accuracy()
            self.val_aupr = torchmetrics.AveragePrecision()
            self.val_auroc = torchmetrics.AUROC()
            self.val_f1 = torchmetrics.F1Score()
            self.metrics = {
                "acc": self.val_accuracy,
                "aupr": self.val_aupr,
                "auroc": self.val_auroc,
                "f1": self.val_f1,
            }
        else:
            self.val_mse = torchmetrics.MeanSquaredError()
            self.val_pcc = torchmetrics.PearsonCorrCoef()
            self.metrics = {"mse": self.val_mse, "pcc": self.val_pcc}

    def forward(self, drug_target_pairs):
        drugs, targets = drug_target_pairs
        drug_projection = self.drug_projector(drugs)
        target_projection = self.target_projector(targets)

        if self.classify:
            similarity = nn.CosineSimilarity()(
                drug_projection, target_projection
            )
        else:
            similarity = torch.bmm(
                drug_projection.view(-1, 1, self.latent_dim),
                target_projection.view(-1, self.latent_dim, 1),
            ).squeeze()

        return similarity

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        drug, protein, label = train_batch
        similarity = self.forward(drug, protein)

        if self.classify:
            sigmoid = torch.nn.Sigmoid()
            similarity = torch.squeeze(sigmoid(similarity))
            loss_fct = torch.nn.BCELoss()
        else:
            loss_fct = torch.nn.MSELoss()

        loss = loss_fct(similarity, label)
        self.log("train/loss", loss)

        return loss

    def validation_step(self, train_batch, batch_idx):
        drug, protein, label = train_batch
        similarity = self.forward(drug, protein)

        if self.classify:
            sigmoid = torch.nn.Sigmoid()
            similarity = torch.squeeze(sigmoid(similarity))
            loss_fct = torch.nn.BCELoss()
        else:
            loss_fct = torch.nn.MSELoss()

        loss = loss_fct(similarity, label)
        self.log("val/loss", loss)
        return {"loss": loss, "preds": similarity, "target": label}

    def validation_step_end(self, outputs):
        for name, metric in self.metrics.items():
            metric(outputs["preds"], outputs["target"])
            self.log(f"val/{name}", metric)
