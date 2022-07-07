import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf

import logging as lg

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def get_config(experiment_id, mol_feat, prot_feat):
    data_cfg = {
        "batch_size": 32,
        "num_workers": 0,
        "precompute": True,
        "mol_feat": mol_feat,
        "prot_feat": prot_feat,
    }
    model_cfg = {
        "latent_size": 1024,
    }
    training_cfg = {
        "n_epochs": 50,
        "every_n_val": 1,
    }
    cfg = {
        "data": data_cfg,
        "model": model_cfg,
        "training": training_cfg,
        "experiment_id": experiment_id,
    }

    return OmegaConf.structured(cfg)


def sigmoid_cosine_distance_p(x, y, p=1):
    sig = torch.nn.Sigmoid()
    cosine_sim = torch.nn.CosineSimilarity()
    return (1 - sig(cosine_sim(x, y))) ** p
