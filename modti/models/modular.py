import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchmetrics
from modti.utils import get_activation
from modti.models.base import BaseTrainer
from modti.models.pred_layers import DeepConcat, MLP
from modti.models.pred_layers import AVAILABLE_PRED_LAYERS
from modti.apps.utils import load_hp, parse_overrides, nested_dict_update
from modti.data import get_dataset, train_val_test_split
import numpy as np
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
        pred_layer_type="DeepConcat",
        **module_layer_params
    ):
        super().__init__()
        self.nb_modules =len(task_dim) # TODO: Replace this with the number of target featurization models we have specified
        self.input_dim = input_dim
        self.task_dim = task_dim
        self.latent_dim = latent_dim
        self.activation = get_activation(activation)
        self.op = op
        pred_layer_class = AVAILABLE_PRED_LAYERS[pred_layer_type]
        self.pred_modules = nn.ModuleList([
            pred_layer_class(input_dim=latent_dim, task_dim=latent_dim, **module_layer_params, output_dim=1 if op else 2)
            for _ in range(self.nb_modules)]) #TODO: Use Cosine similarity for reproducing. MLP might not respect structure

        self.input_projector = []
        self.task_projector = []
        for i in range(self.nb_modules):
            self.task_projector.append(nn.Sequential(nn.Linear(self.task_dim[i], self.latent_dim), self.activation))
        
        self.input_projector = nn.Sequential(nn.Linear(self.input_dim, self.latent_dim), self.activation)
        self.op_layer = MLP(self.input_dim, [1], activation=None)

    def forward(self, input_task_pairs):
        inputs = list(input_task_pairs)[0]
        tasks = list(input_task_pairs)[1:]

        #if len(inputs) != 1:
        #    raise NotImplementedError("More than 1 drug featurizer is not supported yet.")

        input_reprs = []
        task_reprs = []
        
        input_reprs.append(self.input_projector(inputs))

        for i, elem in enumerate(tasks):
            task_reprs.append(self.task_projector[i](elem))

        if self.op:
            outs = [modul(input_reprs[0].reshape(1, self.latent_dim), 
                        task_reprs[i].reshape(1, self.latent_dim)) for i, modul in enumerate(self.pred_modules)]
            outs = torch.cat(outs, dim=-1)
            probs = self.op_layer(input_reprs[0])
        else:
            prob_outs = [torch.Tensor(list(modul(input_reprs[0].reshape(1, self.latent_dim),
                             task_reprs[i].reshape(1, self.latent_dim)))) for i, modul in enumerate(self.pred_modules)]  # list of (n, 2)
            prob_outs = torch.stack(prob_outs, dim=-1)  # (n, 2, M)
            probs, outs = prob_outs[0, :], prob_outs[1, :] # (n, M)
        preds = (probs * outs).sum(-1)
        return F.softmax(preds)


class Modular(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network = ModularNetwork(**self._network_params)


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')# temp solution !!!!
    config = load_hp(conf_path="modti/apps/configs/modular.yaml")

    seed = config.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    train = get_dataset(**config.get("dataset"), datatype="train")
    train.__getitem__(0)

    model = Modular(**config.get("model"), **train.get_model_related_params())
    output = model([x.cpu() for x in list(train.__getitem__(0)[0])])
    print(output)