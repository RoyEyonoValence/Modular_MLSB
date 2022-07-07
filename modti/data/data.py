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


def get_prot_featurizer(name, **params):
    try:
        embedder = bio_emb.name_to_embedder[name]
    except AttributeError:
        raise ValueError(
            f"Specified molecule featurizer {name} is not supported. Options are {list(bio_emb.name_to_embedder.keys())}"
        )
    return embedder


def get_mol_featurizer(name, **params):
    try:
        embedder = functools.partial(datamol.to_fp, fp_type=name, **params)
    except AttributeError:
        raise ValueError(
            f"Specified molecule featurizer {name} is not supported. Options are {list(bio_emb.name_to_embedder.keys())}"
        )
    return embedder


class DTIDataset(Dataset):
    def __init__(self, name, drug_featurizer_params, prot_featurizer_params, **kwargs):
        super(DTIDataset, self).__init__()
        drugs, targets, labels = get_raw_dataset(name, **kwargs)
        self.drugs = drugs
        self.targets = targets
        self.labels = labels
        assert len(drugs) == len(targets)
        assert len(targets) == len(labels)
        self.drug_featurizer = get_mol_featurizer(**drug_featurizer_params)
        self.target_featurizer = get_prot_featurizer(**prot_featurizer_params)

    def __len__(self):
        return len(self.drugs)

    def __getitem__(self, i):
        drug = self.drug_featurizer(self.drugs[i])
        target = self.target_featurizer(self.targets[i])
        label = torch.tensor(self.labels[i])
        return drug, target, label


def get_dataset(*args, **kwargs):
    return DTIDataset(*args, **kwargs)


def train_test_val_split(dataset, val_size=0.2, test_size=0.2):
    assert 0 < val_size < 1, "Train_size should be between 0 and 1"
    assert 0 < test_size < 1, "Test_size should be between 0 and 1"
    dsize = len(dataset)
    test_size = int(dsize * test_size)
    train_size = dsize - test_size

    tmp, test = random_split(dataset, lengths=[train_size, test_size])

    dsize = len(tmp)
    val_size = int(dsize * val_size)
    train_size = len(dataset) - test_size - val_size
    train, val = random_split(tmp, lengths=[train_size, val_size])
    return train, val