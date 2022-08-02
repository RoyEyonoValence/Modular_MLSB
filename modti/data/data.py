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
    mol_emb, prot_emb = zip(*tmp)
    if pad:
        proteins = pad_sequence(prot_emb, batch_first=True)
    else:
        proteins = torch.stack(prot_emb, 0)
    molecules = torch.stack(mol_emb, 0)
    affinities = torch.stack(labels, 0)
    return (molecules, proteins), affinities


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
        assert len(drugs) == len(targets)
        assert len(targets) == len(labels)
        self.drug_featurizer = get_mol_featurizer(**drug_featurizer_params)
        self.target_featurizer = get_target_featurizer(**target_featurizer_params)
        self.drug_featurizer_params = drug_featurizer_params
        self.target_featurizer_params = target_featurizer_params
        self.__mol_emb_size__ = None
        self.__target_emb_size__ = None
        self.precompute_features()

    def __len__(self):
        return len(self.drugs)

    def __getitem__(self, i):
        drug = to_tensor(self.drugs[i], dtype=torch.float32).cuda()
        target = to_tensor(self.targets[i].mean(0), dtype=torch.float32).cuda()
        label = torch.tensor(self.labels[i]).cuda()
        if self.__target_emb_size__ is None:
            self.__target_emb_size__ = target.shape[-1]
        if self.__mol_emb_size__ is None:
            self.__mol_emb_size__ = drug.shape[-1]
        return (drug, target), label

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
    
    def precompute_features(self):
        print("Precomputing drug and protein featurization ...")
        for i in range(len(self.drugs)):
            self.drugs[i] = self.drug_featurizer(self.drugs[i])

        for i in range(len(self.targets)):
            if self.target_featurizer_params['name']=="esm":
                _max_len = 1024
                if len(self.targets[i]) > _max_len - 2:
                    self.targets[i] = self.target_featurizer(self.targets[i][: _max_len - 2])
                else:
                    self.targets[i] = self.target_featurizer(self.targets[i])
            else:
                self.targets[i] = self.target_featurizer(self.targets[i])
        

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