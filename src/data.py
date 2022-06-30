import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import dscript
import os
import pickle as pk
import pandas as pd
import pytorch_lightning as pl

from types import SimpleNamespace
from tqdm import tqdm
from omegaconf import OmegaConf
from functools import lru_cache
from numpy.random import choice
from torch.nn.utils.rnn import pad_sequence
from tdc import utils as tdc_utils
from tdc.benchmark_group import dti_dg_group

from .featurizers import Featurizer
from .utils import get_logger
from pathlib import Path
import typing as T

logg = get_logger()

def get_task_dir(task_name):
    if task_name.lower() == 'biosnap':
        return Path('./dataset/BIOSNAP/full_data').resolve()
    elif task_name.lower() == 'bindingdb':
        return Path('./dataset/BindingDB').resolve()
    elif task_name.lower() == 'davis':
        return Path('./dataset/DAVIS').resolve()
    elif task_name.lower() == 'biosnap_prot':
        return Path('./dataset/BIOSNAP/unseen_protein').resolve()
    elif task_name.lower() == 'biosnap_mol':
        return Path('./dataset/BIOSNAP/unseen_drug').resolve()
    
def drug_target_collate_fn(args, pad=False):
    """
    Collate function for PyTorch data loader.

    :param args: Batch of training samples with molecule, protein, and affinity
    :type args: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    """
    memb = [a[0] for a in args]
    pemb = [a[1] for a in args]
    labs = [a[2] for a in args]

    if pad:
        proteins = pad_sequence(pemb,batch_first=True)
    else:
        proteins = torch.stack(pemb, 0)
    molecules = torch.stack(memb, 0)
    affinities = torch.stack(labs, 0)

    return molecules, proteins, affinities

def make_contrastive(df: pd.DataFrame,
                     drug_column: str,
                     target_column: str,
                     label_column: str,
                     n_neg_per: int = 50,
                    ): 
    pos_df = df[df[label_column] == 1]
    neg_df = df[df[label_column] == 0]
    
    contrastive = []

    for _,r in pos_df.iterrows():
        for _ in range(n_neg_per):
            contrastive.append((r[target_column], r[drug_column], choice(neg_df[drug_column])))

    contrastive = pd.DataFrame(contrastive,columns=['Anchor','Positive','Negative'])
    return contrastive

class BinaryDataset(Dataset):
    def __init__(self,
                 drugs,
                 targets,
                 labels,
                 drug_featurizer: Featurizer,
                 target_featurizer: Featurizer
                ):
        self.drugs = drugs
        self.targets = targets
        self.labels = labels
        
        self.drug_featurizer = drug_featurizer
        self.target_featurizer = target_featurizer

    def __len__(self):
        return len(self.drugs)

    def __getitem__(self, i):
        drug = self.drug_featurizer(self.drugs[i])
        target = self.target_featurizer(self.targets[i])
        label = torch.tensor(self.labels[i])

        return drug, target, label
    
class ContrastiveDataset(Dataset):
    def __init__(self,
                 anchors,
                 positives,
                 negatives,
                 drug_featurizer: Featurizer,
                 target_featurizer: Featurizer
                ):
        self.anchors = anchors
        self.positives = positives
        self.negatives = negatives
        
        self.drug_featurizer = drug_featurizer
        self.target_featurizer = target_featurizer

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, i):
        
        anchorEmb = self.target_featurizer(self.anchors[i])
        positiveEmb = self.drug_featurizer(self.positives[i])
        negativeEmb = self.drug_featurizer(self.negatives[i])

        return anchorEmb, positiveEmb, negativeEmb

class DTIDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 drug_featurizer: Featurizer,
                 target_featurizer: Featurizer,
                 device: torch.device = torch.device("cpu"),
                 batch_size: int = 32,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 header = 0,
                 index_col = 0,
                 sep = ","
                ):
        
        self._loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "collate_fn": drug_target_collate_fn
        }
        
        self._csv_kwargs = {
            "header": header,
            "index_col": index_col,
            "sep": sep
        }
        
        self._device = device
        
        self._data_dir = Path(data_dir)
        self._train_path = Path("train.csv")
        self._val_path = Path("val.csv")
        self._test_path = Path("test.csv")
        
        self._drug_column = "SMILES"
        self._target_column = "Target Sequence"
        self._label_column = "Label"
        
        self.drug_featurizer = drug_featurizer
        self.target_featurizer = target_featurizer
        
    def prepare_data(self):
        
        if self.drug_featurizer.path.exists() and self.target_featurizer.path.exists():
            logg.warning("Drug and target featurizers already exist")
            return
        
        df_train = pd.read_csv(self._data_dir / self._train_path,
                                    **self._csv_kwargs
                                   )
        df_val = pd.read_csv(self._data_dir / self._val_path,
                                  **self._csv_kwargs
                                 )
        df_test = pd.read_csv(self._data_dir / self._test_path,
                                   **self._csv_kwargs
                                  )
        dataframes = [df_train, df_val, df_test]
        all_drugs = pd.concat([i[self._drug_column] for i in dataframes]).unique()
        all_targets = pd.concat([i[self._target_column] for i in dataframes]).unique()
        
        if self._device.type == "cuda":
            self.drug_featurizer.cuda(self._device)
            self.target_featurizer.cuda(self._device)
        
        if not self.drug_featurizer.path.exists():
            self.drug_featurizer.write_to_disk(all_drugs)
            
        if not self.target_featurizer.path.exists():
            self.target_featurizer.write_to_disk(all_targets)
            
        self.drug_featurizer.cpu()
        self.target_featurizer.cpu()
        
    def setup(self, stage: T.Optional[str] = None):
        
        self.df_train = pd.read_csv(self._data_dir / self._train_path,
                                    **self._csv_kwargs
                                   )
        self.df_val = pd.read_csv(self._data_dir / self._val_path,
                                  **self._csv_kwargs
                                 )
        self.df_test = pd.read_csv(self._data_dir / self._test_path,
                                   **self._csv_kwargs
                                  )
        self._dataframes = [self.df_train, self.df_val, self.df_test]
        
        all_drugs = pd.concat([i[self._drug_column] for i in self._dataframes]).unique()
        all_targets = pd.concat([i[self._target_column] for i in self._dataframes]).unique()
        
        if self._device.type == "cuda":
            self.drug_featurizer.cuda(self._device)
            self.target_featurizer.cuda(self._device)
            
        self.drug_featurizer.preload(all_drugs)
        self.drug_featurizer.cpu()
        
        self.target_featurizer.preload(all_targets)
        self.target_featurizer.cpu()
        
        if stage == "fit" or stage is None:    
            self.data_train = BinaryDataset(self.df_train[self._drug_column],
                                            self.df_train[self._target_column],
                                            self.df_train[self._label_column],
                                            self.drug_featurizer,
                                            self.target_featurizer
                                           )
            
            self.data_val = BinaryDataset(self.df_val[self._drug_column],
                                            self.df_val[self._target_column],
                                            self.df_val[self._label_column],
                                            self.drug_featurizer,
                                            self.target_featurizer
                                           )
                
        if stage == "test" or stage is None:
            self.data_test = BinaryDataset(self.df_test[self._drug_column],
                                            self.df_test[self._target_column],
                                            self.df_test[self._label_column],
                                            self.drug_featurizer,
                                            self.target_featurizer
                                           )
            
    def train_dataloader(self):
        return DataLoader(self.data_train, 
                          **self._loader_kwargs
                         )

    def val_dataloader(self):
        return DataLoader(self.data_val,
                        **self._loader_kwargs
                         )

    def test_dataloader(self):
        return DataLoader(self.data_test,
                         **self._loader_kwargs
                         )
    
class TDCDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 drug_featurizer: Featurizer,
                 target_featurizer: Featurizer,
                 device: torch.device = torch.device("cpu"),
                 batch_size: int = 32,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 header = 0,
                 index_col = 0,
                 sep = ","
                ):
        
        self._loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "collate_fn": drug_target_collate_fn
        }
        
        self._csv_kwargs = {
            "header": header,
            "index_col": index_col,
            "sep": sep
        }
        
        self._device = device
        
        self._data_dir = Path(data_dir)
        self._train_path = Path("train.csv")
        self._val_path = Path("val.csv")
        self._test_path = Path("test.csv")
        
        self._drug_column = "SMILES"
        self._target_column = "Target Sequence"
        self._label_column = "Label"
        
        self.drug_featurizer = drug_featurizer
        self.target_featurizer = target_featurizer
        
    def prepare_data(self):
        
        if self.drug_featurizer.path.exists() and self.target_featurizer.path.exists():
            logg.warning("Drug and target featurizers already exist")
            return
        
        df_train = pd.read_csv(self._data_dir / self._train_path,
                                    **self._csv_kwargs
                                   )
        df_val = pd.read_csv(self._data_dir / self._val_path,
                                  **self._csv_kwargs
                                 )
        df_test = pd.read_csv(self._data_dir / self._test_path,
                                   **self._csv_kwargs
                                  )
        dataframes = [df_train, df_val, df_test]
        all_drugs = pd.concat([i[self._drug_column] for i in dataframes]).unique()
        all_targets = pd.concat([i[self._target_column] for i in dataframes]).unique()
        
        if self._device.type == "cuda":
            self.drug_featurizer.cuda(self._device)
            self.target_featurizer.cuda(self._device)
        
        if not self.drug_featurizer.path.exists():
            self.drug_featurizer.write_to_disk(all_drugs)
            
        if not self.target_featurizer.path.exists():
            self.target_featurizer.write_to_disk(all_targets)
            
        self.drug_featurizer.cpu()
        self.target_featurizer.cpu()
        
    def setup(self, stage: T.Optional[str] = None):
        
        self.df_train = pd.read_csv(self._data_dir / self._train_path,
                                    **self._csv_kwargs
                                   )
        self.df_val = pd.read_csv(self._data_dir / self._val_path,
                                  **self._csv_kwargs
                                 )
        self.df_test = pd.read_csv(self._data_dir / self._test_path,
                                   **self._csv_kwargs
                                  )
        self._dataframes = [self.df_train, self.df_val, self.df_test]
        
        all_drugs = pd.concat([i[self._drug_column] for i in self._dataframes]).unique()
        all_targets = pd.concat([i[self._target_column] for i in self._dataframes]).unique()
        
        if self._device.type == "cuda":
            self.drug_featurizer.cuda(self._device)
            self.target_featurizer.cuda(self._device)
            
        self.drug_featurizer.preload(all_drugs)
        self.drug_featurizer.cpu()
        
        self.target_featurizer.preload(all_targets)
        self.target_featurizer.cpu()
        
        if stage == "fit" or stage is None:    
            self.data_train = BinaryDataset(self.df_train[self._drug_column],
                                            self.df_train[self._target_column],
                                            self.df_train[self._label_column],
                                            self.drug_featurizer,
                                            self.target_featurizer
                                           )
            
            self.data_val = BinaryDataset(self.df_val[self._drug_column],
                                            self.df_val[self._target_column],
                                            self.df_val[self._label_column],
                                            self.drug_featurizer,
                                            self.target_featurizer
                                           )
                
        if stage == "test" or stage is None:
            self.data_test = BinaryDataset(self.df_test[self._drug_column],
                                            self.df_test[self._target_column],
                                            self.df_test[self._label_column],
                                            self.drug_featurizer,
                                            self.target_featurizer
                                           )
            
    def train_dataloader(self):
        return DataLoader(self.data_train, 
                          **self._loader_kwargs
                         )

    def val_dataloader(self):
        return DataLoader(self.data_val,
                        **self._loader_kwargs
                         )

    def test_dataloader(self):
        return DataLoader(self.data_test,
                         **self._loader_kwargs
                         )
    
class DUDEDataModule(pl.LightningDataModule):
    def __init__(self,
                 contrastive_split: str,
                 drug_featurizer: Featurizer,
                 target_featurizer: Featurizer,
                 device: torch.device = torch.device("cpu"),
                 n_neg_per: int = 50,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 header = 0,
                 index_col = None,
                 sep = "\t"
                ):
        
        self._loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "collate_fn": drug_target_collate_fn
        }
        
        self._csv_kwargs = {
            "header": header,
            "index_col": index_col,
            "sep": sep
        }
        
        self._device = device
        self._n_neg_per = n_neg_per
        
        self._data_dir = Path("./dataset/DUDe/")
        self._split = contrastive_split
        self._split_path = self._data_dir / Path(f"dude_{self._split}_type_train_test_split.csv")
        
        self._drug_id_column = "Molecule_ID"
        self._drug_column = "Molecule_SMILES"
        self._target_id_column = "Target_ID"
        self._target_column = "Target_Seq"
        self._label_column = "Label"
        
        self.drug_featurizer = drug_featurizer
        self.target_featurizer = target_featurizer
        
    def setup(self, stage: T.Optional[str] = None):
        
        self.df_full = pd.read_csv(self._data_dir / Path("full.tsv"),
                                  **self._csv_kwargs
                                  )
        
        self.df_splits = pd.read_csv(self._split_path, header=None)
        self._train_list = self.df_splits[self.df_splits[1] == "train"][0].values
        self._test_list = self.df_splits[self.df_splits[1] == "test"][0].values
        
        self.df_train = self.df_full[self.df_full[self._target_id_column].isin(self._train_list)]
        self.df_test = self.df_full[self.df_full[self._target_id_column].isin(self._test_list)]
        
        self.train_contrastive = make_contrastive(self.df_train,
                                               self._drug_column,
                                               self._target_column,
                                               self._label_column,
                                               self._n_neg_per
                                              )
        
        self._dataframes = [self.df_train]#, self.df_test]
        
        all_drugs = pd.concat([i[self._drug_column] for i in self._dataframes]).unique()
        all_targets = pd.concat([i[self._target_column] for i in self._dataframes]).unique()
        
        if self._device.type == "cuda":
            self.drug_featurizer.cuda(self._device)
            self.target_featurizer.cuda(self._device)
            
        self.drug_featurizer.precompute(all_drugs)
        self.drug_featurizer.cpu()
        
        self.target_featurizer.precompute(all_targets)
        self.target_featurizer.cpu()
        
        if stage == "fit" or stage is None:    
            self.data_train = ContrastiveDataset(self.train_contrastive["Anchor"],
                                                self.train_contrastive["Positive"],
                                                self.train_contrastive["Negative"],
                                                self.drug_featurizer,
                                                self.target_featurizer
                                               )
            
        # if stage == "test" or stage is None:
        #     self.data_test = BinaryDataset(self.df_test[self._drug_column],
        #                                     self.df_test[self._target_column],
        #                                     self.df_test[self._label_column],
        #                                     self.drug_featurizer,
        #                                     self.target_featurizer
        #                                    )
            
    def train_dataloader(self):
        return DataLoader(self.data_train, 
                          **self._loader_kwargs
                         )

#     def val_dataloader(self):
#         return DataLoader(self.data_test,
#                         **self._loader_kwargs
#                          )

#     def test_dataloader(self):
#         return DataLoader(self.data_test,
#                          **self._loader_kwargs
#                          )