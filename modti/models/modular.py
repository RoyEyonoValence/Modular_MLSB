import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchmetrics
from traitlets import Integer
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
        self.input_dim = input_dim if isinstance(input_dim, list) else [input_dim]
        self.task_dim = task_dim if isinstance(task_dim, list) else [task_dim]
        self.nb_modules =len(self.task_dim) * len(self.input_dim) # TODO: Replace this with the number of target featurization models we have specified
        self.latent_dim = latent_dim
        self.activation = get_activation(activation)
        self.op = op
        pred_layer_class = AVAILABLE_PRED_LAYERS[pred_layer_type]
        self.pred_modules = nn.ModuleList([
            pred_layer_class(input_dim=latent_dim, task_dim=latent_dim, **module_layer_params, output_dim=1 if op else 2)
            for _ in range(self.nb_modules)]) #TODO: Use Cosine similarity for reproducing. MLP might not respect structure

        self.input_projector = []

        self.task_projector = nn.ModuleList([nn.Sequential(nn.Linear(self.task_dim[i], self.latent_dim), self.activation) 
                                            for i in range(len(self.task_dim))])

        self.input_projector = nn.ModuleList([nn.Sequential(nn.Linear(self.input_dim[i], self.latent_dim), self.activation) 
                                            for i in range(len(self.input_dim))])
                                            
        # self.op_layer = MLP(self.input_dim, [1], activation=None)

    def forward(self, input_task_pairs):
        inputs, tasks = input_task_pairs
        # inputs = list(input_task_pairs)[0]
        # tasks = list(input_task_pairs)[1:]

        #if len(inputs) != 1:
        #    raise NotImplementedError("More than 1 drug featurizer is not supported yet.")
        sum_modules = 0
        index = 0

        for i, molecule in enumerate(inputs):
            for j, protein in enumerate(tasks):

                sum_modules += self.pred_modules[index](self.input_projector[i](molecule),
                                                     self.task_projector[j](protein))[0]*self.pred_modules[index](self.input_projector[i](molecule),
                                                      self.task_projector[j](protein))[1]
                index += 1


        #preds = self.predict([self.input_projector(inputs)], [self.task_projector[i](task) for i, task in enumerate(tasks)])
        
        return F.sigmoid(sum_modules)

    def predict(self, input, tasks):

        

        if self.op:
            raise NotImplementedError("Modular op is not supported yet.")
            outs = [modul(input[0].reshape(1, self.latent_dim), 
                        tasks[i].reshape(1, self.latent_dim)) for i, modul in enumerate(self.pred_modules)]
            outs = torch.cat(outs, dim=-1)
            probs = self.op_layer(input[0])
        else:
            
            prob_outs = [self.pred_modules[i](input[0], tasks[i])[0]*self.pred_modules[i](input[0], tasks[i])[1]
                            for i in range(len(self.pred_modules))] # list of (n, 2)

        return sum(prob_outs)



class Modular(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network = ModularNetwork(**self._network_params)


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')# temp solution !!!!
    torch.multiprocessing.set_start_method('spawn')
    config = load_hp(conf_path="modti/apps/configs/modular.yaml")

    seed = config.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    train = get_dataset(**config.get("dataset"), datatype="train")
    train.__getitem__(0)

    model = Modular(**config.get("model"), **train.get_model_related_params())

    params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6,
          'collate_fn': train.collate_fn}

    training_generator = torch.utils.data.DataLoader(train, **params)

    for batch, labels in training_generator:
        output = model(batch)
        break
    print(output)