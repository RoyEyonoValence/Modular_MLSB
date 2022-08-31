from torch import nn
from modti.utils import get_activation
from modti.models.base import BaseTrainer
from modti.models.pred_layers import AVAILABLE_PRED_LAYERS
import torch
from modti.apps.utils import load_hp, parse_overrides, nested_dict_update
from modti.data import get_dataset, train_val_test_split
import numpy as np


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
        self.pred_layer = pred_layer_class(input_dim=latent_dim, task_dim=latent_dim, **pred_layer_params, output_dim=1)

    def forward(self, input_task_pairs):
        inputs, tasks = input_task_pairs
        input_reprs = self.input_projector(inputs)
        task_reprs = self.task_projector(tasks)
        return self.pred_layer(input_reprs, task_reprs)


class Monolithic(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network = MonolithicNetwork(**self._network_params)


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')# temp solution !!!!
    torch.multiprocessing.set_start_method('spawn')
    config = load_hp(conf_path="modti/apps/configs/monolithic_mini.yaml")

    seed = config.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    train = get_dataset(**config.get("dataset"), datatype="train")
    train.__getitem__(0)

    model = Monolithic(**config.get("model"), **train.get_model_related_params())

    params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6,
          'collate_fn': train.collate_fn}

    training_generator = torch.utils.data.DataLoader(train, **params)

    for batch, labels in training_generator:
        output = model([x.cpu() for x in batch])
        break
    print(output)