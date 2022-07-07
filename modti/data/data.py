import torch
import datamol
import functools
import typing as T
import pandas as pd
import pytorch_lightning as pl
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, random_split
from bio_embeddings import embed as bio_emb


DATASET_DIRECTORIES = dict(
    biosnap="./dataset/BIOSNAP/full_data",
    bindingdb="./dataset/BindingDB",
    davis="./dataset/DAVIS",
    biosnap_prot="./dataset/BIOSNAP/unseen_protein",
    biosnap_mol="./dataset/BIOSNAP/unseen_drug"
)


def get_dataset_dir(name):
    if name in DATASET_DIRECTORIES:
        return Path(DATASET_DIRECTORIES[name]).resolve()
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
    return df[_drug_column].data, df[_target_column].data, df[_label_column].data


def drug_target_collate_fn(args, pad=False):
    """
    Collate function for PyTorch data loader.
    :param args: Batch of training samples with molecule, protein, and affinity
    :type args: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    """
    mol_emb, prot_emb, labels = zip(*args)

    if pad:
        proteins = pad_sequence(prot_emb, batch_first=True)
    else:
        proteins = torch.stack(prot_emb, 0)
    molecules = torch.stack(mol_emb, 0)
    affinities = torch.stack(labels, 0)
    return molecules, proteins, affinities


def get_target_featurizer(name, **params):
    try:
        embedder = bio_emb.name_to_embedder[name]
    except AttributeError:
        raise ValueError(
            f"Specified Target featurizer {name} is not supported. Options are {list(bio_emb.name_to_embedder.keys())}"
        )
    return embedder


def get_mol_featurizer(name, **params):
    try:
        embedder = functools.partial(datamol.to_fp, fp_type=name, **params)
    except AttributeError:
        raise ValueError(
            f"Specified molecule featurizer {name} is not supported. Options are {datamol.list_supported_fingerprints()}"
        )
    return embedder


class DTIDataset(Dataset):
    def __init__(self, name, drug_featurizer_params, target_featurizer_params, **kwargs):
        super(DTIDataset, self).__init__()
        drugs, targets, labels = get_raw_dataset(name, **kwargs)
        self.drugs = drugs
        self.targets = targets
        self.labels = labels
        assert len(drugs) == len(targets)
        assert len(targets) == len(labels)
        self.drug_featurizer = get_mol_featurizer(**drug_featurizer_params)
        self.target_featurizer = get_target_featurizer(**target_featurizer_params)
        self.__mol_emb_size__ = None
        self.__target_emb_size__ = None

    def __len__(self):
        return len(self.drugs)

    def __getitem__(self, i):
        drug = self.drug_featurizer(self.drugs[i])
        target = self.target_featurizer(self.targets[i])
        label = torch.tensor(self.labels[i])
        if self.__prot_emb_size__ is None:
            self.__prot_emb_size__ = target.shape[-1]
        if self.__mol_emb_size__ is None:
            self.__mol_emb_size__ = drug.shape[-1]
        return drug, target, label

    @property
    def mol_embedding_size(self):
        if self.__mol_emb_size__ is None:
            _ = self.__getitem__(0)
        return self.__mol_emb_size__

    @property
    def prot_embedding_size(self):
        if self.__prot_emb_size__ is None:
            _ = self.__getitem__(0)
        return self.__prot_emb_size__

    def get_embedding_sizes(self):
        return dict(mol_embedding_size=self.mol_embedding_size, prot_embedding_size=self.prot_embedding_size)


def get_dataset(*args, **kwargs):
    return DTIDataset(*args, **kwargs)


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
    return train, val