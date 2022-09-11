import os
import torch
import datamol
import functools
import pandas as pd
from pathlib import Path
from sklearn.utils.multiclass import type_of_target
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, random_split
from bio_embeddings import embed as bio_emb
from modti.utils import parent_at_depth, to_tensor
import numpy as np
from modti.apps.utils import load_hp, parse_overrides, nested_dict_update


dataset_dir = os.path.join(parent_at_depth(__file__, 3), "artifacts/datasets")

DATASET_DIRECTORIES = dict(
    biosnap="BIOSNAP/full_data",
    bindingdb="BindingDB",
    davis="DAVIS",
    biosnap_prot="BIOSNAP/unseen_protein",
    biosnap_mol="BIOSNAP/unseen_drug"
)


def get_dataset_dir(name):
    if name in DATASET_DIRECTORIES:
        return Path(os.path.join(dataset_dir, DATASET_DIRECTORIES[name])).resolve()
    else:
        raise ValueError("Unknown Dataset type")


def get_raw_dataset(name, **_csv_kwargs):
    path = get_dataset_dir(name)
    _train_path = Path("train.csv")
    _val_path = Path("val.csv")
    _test_path = Path("test.csv")

    _drug_column = "SMILES"
    _target_column = "Target Sequence"
    _label_column = "Label"
    df_train = pd.read_csv(path / _train_path, **_csv_kwargs)
    df_val = pd.read_csv(path / _val_path, **_csv_kwargs)
    df_test = pd.read_csv(path / _test_path, **_csv_kwargs)
    df = pd.concat([df_train, df_val, df_test], axis=0)

    return df[_drug_column].values, df[_target_column].values, df[_label_column].values



def get_raw_dataset_split(name, datatype, **_csv_kwargs):
    path = get_dataset_dir(name)
    _train_path = Path("train.csv")
    _val_path = Path("val.csv")
    _test_path = Path("test.csv")

    _drug_column = "SMILES"
    _target_column = "Target Sequence"
    _label_column = "Label"

    if datatype == "train":
        df = pd.read_csv(path / _train_path, **_csv_kwargs)
    elif datatype == "val":
        df = pd.read_csv(path / _val_path, **_csv_kwargs)
    else:
        df = pd.read_csv(path / _test_path, **_csv_kwargs)

    return df[_drug_column].values, df[_target_column].values, df[_label_column].values


def dti_collate_fn(args, pad=False):
    """
    Collate function for the data loader.
    :param args: Batch of training samples with molecule, protein, and affinity
    :type args: Iterable[Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]
    """
    tmp, labels = zip(*args)
    affinities = torch.stack(labels, 0)
    # tmp, = tmp if len(tmp) == 1 else tmp
    drugs, targets = zip(*tmp)
    drugs_embeddings = []
    targets_embeddings = []

    # mol_emb, prot_emb = zip(*tmp)

    for i, embed in enumerate(list(zip(*drugs))):
        drugs_embeddings.append(torch.stack(embed, 0))

    for i, embed in enumerate(list(zip(*targets))):
        targets_embeddings.append(torch.stack(embed, 0))

    # tmp = torch.stack(tmp, 0)

    '''molecules = torch.stack(mol_emb, 0)

    if pad:
        proteins = pad_sequence(prot_emb, batch_first=True)
    else:
        proteins = torch.stack(prot_emb, 0)'''

    if len(drugs_embeddings) == 1 and len(targets_embeddings) == 1:
        return tuple([drugs_embeddings[0], targets_embeddings[0]]), affinities

    
    return tuple([drugs_embeddings, targets_embeddings]), affinities


def get_target_featurizer(name, **params):

    

    try:
        embedder = bio_emb.name_to_embedder[name](**params, device='cuda').embed
    except:
        raise ValueError(
            f"Specified Target featurizer {name} is not supported. Options are {list(bio_emb.name_to_embedder.keys())}"
        )

    return embedder


def get_mol_featurizer(name, **params):
    try:
        embedder = functools.partial(datamol.to_fp, fp_type=name, **params)
    except:
        raise ValueError(
            f"Specified molecule featurizer {name} is not supported. Options are {datamol.list_supported_fingerprints()}"
        )
    return embedder


class DTIDataset(Dataset):
    def __init__(self, name, drug_featurizer_params, target_featurizer_params, datatype=None, **kwargs):
        super(DTIDataset, self).__init__()
        if datatype is not None:
            drugs, targets, labels = get_raw_dataset_split(name, datatype, **kwargs)
        else:
            drugs, targets, labels = get_raw_dataset(name, **kwargs)
        self.drugs = drugs
        self.targets = targets
        self.labels = labels
        self.featurized_targets = []
        self.featurized_drugs = []
        assert len(drugs) == len(targets)
        assert len(targets) == len(labels)
        self.drug_featurizer = {}
        self.target_featurizer = {}
        for featurizer in drug_featurizer_params['name']:
            self.drug_featurizer[featurizer] = get_mol_featurizer(featurizer)
        
        for featurizer in target_featurizer_params['name']:
            self.target_featurizer[featurizer] = get_target_featurizer(featurizer) #TODO: This will be a list when it is modular
        
        self.drug_featurizer_params = drug_featurizer_params
        self.target_featurizer_params = target_featurizer_params #TODO: This will be a list when it is modular
        self.__mol_emb_size__ = None
        self.__target_emb_size__ = None
        for featurizer_target_name  in self.target_featurizer:
            _, target = self.precompute_features(drug=self.drugs,target=self.targets,
                                                featurizer_target = (featurizer_target_name,
                                                 self.target_featurizer[featurizer_target_name]))
            self.featurized_targets.append(target)
        
        for featurizer_drug_name  in self.drug_featurizer:
            drug, _ = self.precompute_features(drug=self.drugs, target=self.targets,
                                                featurizer_drug = (featurizer_drug_name,
                                                 self.drug_featurizer[featurizer_drug_name]))
            self.featurized_drugs.append(drug)

    def __len__(self):
        return len(self.drugs)

    def __getitem__(self, i):
        drugs = []
        targets = []
        for drug in self.featurized_drugs:
            drugs.append(to_tensor(drug[i], dtype=torch.float32).cuda())

        for target in self.featurized_targets:
            targets.append(to_tensor(target[i].mean(0), dtype=torch.float32).cuda())
        

        label = torch.tensor(int(self.labels[i])).cuda()
        if self.__target_emb_size__ is None:
            self.__target_emb_size__ = [targ.shape[-1] for targ in targets]
            if len(self.__target_emb_size__) == 1:
                self.__target_emb_size__ = self.__target_emb_size__[0]
        if self.__mol_emb_size__ is None:
            self.__mol_emb_size__ = [drg.shape[-1] for drg in drugs]
            if len(self.__mol_emb_size__) == 1:
                self.__mol_emb_size__ = self.__mol_emb_size__[0]

        # return tuple(drug_target), label # TODO: The final output of this should be a tuple of different featurizers
        return  tuple([drugs, targets]), label

    @property
    def mol_embedding_size(self):
        if self.__mol_emb_size__ is None:
            _ = self.__getitem__(0)
        return self.__mol_emb_size__

    @property
    def target_embedding_size(self):
        if self.__target_emb_size__ is None:
            _ = self.__getitem__(0)
        return self.__target_emb_size__

    def get_model_related_params(self):
        return dict(
            input_dim=self.mol_embedding_size,
            task_dim=self.target_embedding_size,
            label_type=self.label_type,
            label_dim=self.label_dim,
            collate_fn=self.collate_fn
        )
    
    def precompute_features(self, drug, target, featurizer_target=None, featurizer_drug=None ):
        print("Precomputing drug and protein featurization ...")

        drugs = None
        targets = None
        unique_targets = dict()
        unique_drug = dict()
        if featurizer_drug is not None:
            drugs = []
            featurizer_drug_name, featurize_drug = featurizer_drug
            for i in range(len(drug)):
                try:
                    if drug[i] in unique_drug:
                        drugs.append(unique_drug[drug[i]])
                    else:
                        unique_drug[drug[i]] = featurize_drug(drug[i])
                        drugs.append(unique_drug[drug[i]])
                except:
                    print(f"It seems like the input molecule '{drug[i]}' is invalid.")

        
        if featurizer_target is not None:
            targets = []
            featurizer_target_name, featurize_target = featurizer_target
            for i in range(len(target)):

                if target[i] in unique_targets:
                    targets.append(unique_targets[target[i]])
                else:
                    if featurizer_target_name=="esm":
                        _max_len = 1024
                        if len(target[i]) > _max_len - 2:
                            unique_targets[target[i]] = featurize_target(target[i][: _max_len - 2])
                            targets.append(unique_targets[target[i]])
                        else:
                            unique_targets[target[i]] =  featurize_target(target[i])
                            targets.append(unique_targets[target[i]])
                    else:
                        unique_targets[target[i]] =  featurize_target(target[i])
                        targets.append(unique_targets[target[i]])

        return drugs, targets

        

    @property
    def collate_fn(self):
        return dti_collate_fn

    @property
    def label_type(self):
        if "continuous" in type_of_target(self.labels):
            t = "rgr"
        elif "binary" == type_of_target(self.labels):
            t = "b_clf"
        elif "multiclass" == type_of_target(self.labels):
            t = "mc_clf"
        else:
            raise RuntimeError("Unknown label type")
        return t

    @property
    def label_dim(self):
        return 1 if self.labels.ndim < 2 else self.labels.shape[-1]


def get_dataset(*args, **kwargs):
    return DTIDataset(*args, **kwargs) #TODO: Make sure that this returns the entire dataset


def train_val_test_split(dataset, val_size=0.2, test_size=0.2):
    assert 0 < val_size < 1, "Train_size should be between 0 and 1"
    assert 0 < test_size < 1, "Test_size should be between 0 and 1"
    dsize = len(dataset)
    test_size = int(dsize * test_size)
    tmp_size = dsize - test_size
    tmp, test = random_split(dataset, lengths=[tmp_size, test_size])

    dsize = len(tmp)
    val_size = int(dsize * val_size)
    train_size = dsize - val_size
    train, val = random_split(tmp, lengths=[train_size, val_size])
    return train, val, test # TODO: Inspect split of dataset w.r.t. model evaluation


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')# temp solution !!!!
    config = load_hp(conf_path="modti/apps/configs/modular.yaml")

    seed = config.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # train, valid, test = train_val_test_split(dataset, val_size=0.2, test_size=0.2)
    train = get_dataset(**config.get("dataset"), datatype="train")
    train.__getitem__(0)