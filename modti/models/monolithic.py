from torch import nn
from modti.utils import get_activation
from modti.models.base import BaseTrainer
from modti.models.pred_layers import AVAILABLE_PRED_LAYERS


class MonolithicNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        task_dim,
        label_dim,
        latent_dim=512,
        activation='ReLU',
        pred_layer_type="DeepConcat",
        **pred_layer_params
    ):
        super().__init__()

        self.input_dim = input_dim
        self.task_dim = task_dim
        self.latent_dim = label_dim
        self.latent_dim = latent_dim
        self.activation = get_activation(activation)
        self.input_projector = nn.Sequential(nn.Linear(self.input_dim, self.latent_dim), self.activation)
        self.task_projector = nn.Sequential(nn.Linear(self.task_dim, self.latent_dim), self.activation)
        pred_layer_class = AVAILABLE_PRED_LAYERS[pred_layer_type]
        self.pred_layer = pred_layer_class(input_dim=input_dim, task_dim=task_dim, **pred_layer_params, output_dim=1)

    def forward(self, input_task_pairs):
        inputs, tasks = input_task_pairs
        input_reprs = self.input_projector(inputs)
        task_reprs = self.task_projector(tasks)
        return self.pred_layer(input_reprs, task_reprs)


class Monolithic(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network = MonolithicNetwork(**self._network_params)