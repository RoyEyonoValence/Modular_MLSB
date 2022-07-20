import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchmetrics
from modti.utils import get_activation
from modti.models.base import BaseTrainer
from modti.models.pred_layers import DeepConcat, MLP
import pdb


class ModularNetwork(nn.Module):
    def __init__(
        self,
        nb_modules,
        input_dim,
        task_dim,
        latent_dim=512,
        activation='ReLU',
        op=False,
        **module_layer_params
    ):
        super().__init__()
        self.nb_modules =nb_modules
        self.input_dim = input_dim
        self.task_dim = task_dim
        self.latent_dim = latent_dim
        self.activation = get_activation(activation)
        self.op = op
        self.pred_modules = nn.ModuleList([
            DeepConcat(input_dim=latent_dim, task_dim=latent_dim, **module_layer_params, output_dim=1 if op else 2)
            for _ in range(self.nb_modules)]) #TODO: Use Cosine similarity for reproducing. MLP might not respect structure

        self.input_projector = nn.Sequential(nn.Linear(self.input_dim, self.latent_dim), self.activation)
        self.task_projector = nn.Sequential(nn.Linear(self.task_dim, self.latent_dim), self.activation) #TODO: Consider using activation for Cosine Similarity
        self.op_layer = MLP(self.task_dim, [1], activation=None)

    def forward(self, input_task_pairs):
        inputs, tasks = input_task_pairs
        input_reprs = self.input_projector(inputs)
        task_reprs = self.task_projector(tasks)
        if self.op:
            outs = [modul(input_reprs, task_reprs) for i, modul in enumerate(self.pred_modules)]
            outs = torch.cat(outs, dim=-1)
            probs = self.op_layer(task_reprs)
        else:
            prob_outs = [modul(input_reprs, task_reprs) for i, modul in enumerate(self.pred_modules)]  # list of (n, 2)
            prob_outs = torch.stack(prob_outs, dim=-1)  # (n, 2, M)
            probs, outs = prob_outs[:, 0, :], prob_outs[:, 1, :] # (n, M)
        preds = (F.softmax(probs, dim=-1) * outs).sum(-1)
        return preds


class Modular(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network = ModularNetwork(**self._network_params)

