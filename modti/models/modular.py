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
from chembag.nn.layers.agg import FeatureAgg, InstanceAgg
import wandb

class HadamardPool1D(nn.Module):

    def __init__(self, dim=-2):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return torch.prod(x, dim=self.dim)
class FeatureAggWrapper(nn.Module):
    r"""
    Global Average pooling of a Tensor over one dimension
    Args:
        dim (int, optional): The dimension on which the pooling operation is applied.
    """

    def __init__(self, estimator, pool="mean", input_dim=128, dim=1):
        super().__init__()
        self.estimator = estimator

        if pool == "hadamard":
            self.pool = HadamardPool1D()
        else:
            featureAgg = FeatureAgg(estimator, pool, input_dim, dim)
            self.pool = featureAgg.pool

    def forward(self, x):
        """
        Args:
            x (torch.FloatTensor): Input tensor of size (B, N, D)
        Returns:
            out (torch.FloatTensor): Output tensor of size (B, O)
        """
        out = self.pool(x)
        return self.estimator(out)


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
        self.output_dim_estimator = 512
        #pred_layer_class = AVAILABLE_PRED_LAYERS["DoubleHeadedSimilarity"]
        #self.pred_modules = nn.ModuleList([
        #    pred_layer_class(input_dim=latent_dim, task_dim=latent_dim, **module_layer_params, output_dim=1 if op else 2)
        #    for _ in range(self.nb_modules)]) #TODO: Use Cosine similarity for reproducing. MLP might not respect structure

        # self.input_projector = []

        self.task_projector = nn.ModuleList([nn.Sequential(nn.Linear(self.task_dim[i], self.latent_dim), self.activation) 
                                            for i in range(len(self.task_dim))])

        self.input_projector = nn.ModuleList([nn.Sequential(nn.Linear(self.input_dim[i], self.latent_dim), self.activation) 
                                            for i in range(len(self.input_dim))])
                                            
        self.estimator = [nn.Sequential(nn.Linear(self.latent_dim, self.output_dim_estimator)) for _ in range(self.nb_modules)]
        #self.featagg = FeatureAgg(self.estimator, pool="attn", input_dim=self.latent_dim)

        self.featagg = nn.ModuleList([ FeatureAggWrapper(self.estimator[i],
                                     pool=pred_layer_type, input_dim=self.latent_dim)
                                    for i in range(self.nb_modules)])
        
        self.instanceagg = InstanceAgg(nn.Linear(self.output_dim_estimator, 1), pool="mil-attn")
        # self.op_layer = MLP(self.input_dim, [1], activation=None)

    def forward(self, input_task_pairs):
        inputs, tasks = input_task_pairs
        index = 0
        # self.alphas = torch.Tensor([]).cuda() # Bad Coding
        self.ys = torch.Tensor([]).cuda() # Bad Coding
        res = None

        for i, molecule in enumerate(inputs):
            for j, protein in enumerate(tasks):
                mol_proj = self.input_projector[i](molecule)
                prot_proj = self.task_projector[j](protein)
                mol_prot_proj = torch.stack((mol_proj,prot_proj), dim=-2)
                pred = self.featagg[index](mol_prot_proj)
                # alpha = pred[:,0].unsqueeze(dim=-1)
                y = pred.unsqueeze(dim=-2)
                # self.alphas = torch.cat((self.alphas, F.sigmoid(alpha)), dim=1)
                self.ys = torch.cat((self.ys, y), dim=1)
                index += 1


        #self.alphas = F.softmax(self.alphas)
        # res = (self.alphas * self.ys).sum(dim=-1)
        # res = self.ys.sum(dim=-1)
        # self.ys = self.ys
        res, contrib = self.instanceagg(self.ys, return_contribution=True)
        res = res.squeeze(dim=-1)
        contrib = contrib.squeeze(dim=-1)

        contrib = torch.mean(contrib, dim=0)

        for index, module_contrib in enumerate(contrib):
            name = 'module '+ str(index)
            wandb.log({name: module_contrib})

        return F.sigmoid(res)

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