import torch
import datamol
import functools
import typing as T
import pandas as pd
import pytorch_lightning as pl
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from bio_embeddings import embed as bio_emb

print(dict(bio_emb))
exit()
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


def get_dataset(name, **_csv_kwargs):
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
        mol_feats = getattr(bio_emb, name)()
    except AttributeError:
        raise ValueError(
            f"Specified molecule featurizer {name} is not supported. Options are {bio_emb.}"
        )

class DTIDataset(Dataset):
    def __init__(self, drugs, targets, labels, drug_featurizer_name, target_featurizer):
        self.drugs = drugs
        self.targets = targets
        self.labels = labels
        assert len(drugs) == len(targets)
        assert len(targets) == len(labels)
        self.drug_featurizer = functools.partial(datamol.to_fp(fp_type=drug_featurizer_name, )
        self.target_featurizer = get

    def __len__(self):
        return len(self.drugs)

    def __getitem__(self, i):
        drug = self.drug_featurizer(self.drugs[i])
        target = self.target_featurizer(self.targets[i])
        label = torch.tensor(self.labels[i])
        return drug, target, label


class DTIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        drug_featurizer,
        target_featurizer,
        device: torch.device = torch.device("cpu"),
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        header=0,
        index_col=0,
        sep=",",
    ):

        self._loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "collate_fn": drug_target_collate_fn,
        }

        self._csv_kwargs = {
            "header": header,
            "index_col": index_col,
            "sep": sep,
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

        if (
            self.drug_featurizer.path.exists()
            and self.target_featurizer.path.exists()
        ):
            logg.warning("Drug and target featurizers already exist")
            return

        df_train = pd.read_csv(
            self._data_dir / self._train_path, **self._csv_kwargs
        )
        df_val = pd.read_csv(
            self._data_dir / self._val_path, **self._csv_kwargs
        )
        df_test = pd.read_csv(
            self._data_dir / self._test_path, **self._csv_kwargs
        )
        dataframes = [df_train, df_val, df_test]
        all_drugs = pd.concat(
            [i[self._drug_column] for i in dataframes]
        ).unique()
        all_targets = pd.concat(
            [i[self._target_column] for i in dataframes]
        ).unique()

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

        self.df_train = pd.read_csv(
            self._data_dir / self._train_path, **self._csv_kwargs
        )
        self.df_val = pd.read_csv(
            self._data_dir / self._val_path, **self._csv_kwargs
        )
        self.df_test = pd.read_csv(
            self._data_dir / self._test_path, **self._csv_kwargs
        )
        self._dataframes = [self.df_train, self.df_val, self.df_test]

        all_drugs = pd.concat(
            [i[self._drug_column] for i in self._dataframes]
        ).unique()
        all_targets = pd.concat(
            [i[self._target_column] for i in self._dataframes]
        ).unique()

        if self._device.type == "cuda":
            self.drug_featurizer.cuda(self._device)
            self.target_featurizer.cuda(self._device)

        self.drug_featurizer.preload(all_drugs)
        self.drug_featurizer.cpu()

        self.target_featurizer.preload(all_targets)
        self.target_featurizer.cpu()

        if stage == "fit" or stage is None:
            self.data_train = BinaryDataset(
                self.df_train[self._drug_column],
                self.df_train[self._target_column],
                self.df_train[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

            self.data_val = BinaryDataset(
                self.df_val[self._drug_column],
                self.df_val[self._target_column],
                self.df_val[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

        if stage == "test" or stage is None:
            self.data_test = BinaryDataset(
                self.df_test[self._drug_column],
                self.df_test[self._target_column],
                self.df_test[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

    def train_dataloader(self):
        return DataLoader(self.data_train, **self._loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.data_val, **self._loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.data_test, **self._loader_kwargs)


class TDCDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        drug_featurizer: Featurizer,
        target_featurizer: Featurizer,
        device: torch.device = torch.device("cpu"),
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        header=0,
        index_col=0,
        sep=",",
    ):

        self._loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "collate_fn": drug_target_collate_fn,
        }

        self._csv_kwargs = {
            "header": header,
            "index_col": index_col,
            "sep": sep,
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

        if (
            self.drug_featurizer.path.exists()
            and self.target_featurizer.path.exists()
        ):
            logg.warning("Drug and target featurizers already exist")
            return

        df_train = pd.read_csv(
            self._data_dir / self._train_path, **self._csv_kwargs
        )
        df_val = pd.read_csv(
            self._data_dir / self._val_path, **self._csv_kwargs
        )
        df_test = pd.read_csv(
            self._data_dir / self._test_path, **self._csv_kwargs
        )
        dataframes = [df_train, df_val, df_test]
        all_drugs = pd.concat(
            [i[self._drug_column] for i in dataframes]
        ).unique()
        all_targets = pd.concat(
            [i[self._target_column] for i in dataframes]
        ).unique()

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

        self.df_train = pd.read_csv(
            self._data_dir / self._train_path, **self._csv_kwargs
        )
        self.df_val = pd.read_csv(
            self._data_dir / self._val_path, **self._csv_kwargs
        )
        self.df_test = pd.read_csv(
            self._data_dir / self._test_path, **self._csv_kwargs
        )
        self._dataframes = [self.df_train, self.df_val, self.df_test]

        all_drugs = pd.concat(
            [i[self._drug_column] for i in self._dataframes]
        ).unique()
        all_targets = pd.concat(
            [i[self._target_column] for i in self._dataframes]
        ).unique()

        if self._device.type == "cuda":
            self.drug_featurizer.cuda(self._device)
            self.target_featurizer.cuda(self._device)

        self.drug_featurizer.preload(all_drugs)
        self.drug_featurizer.cpu()

        self.target_featurizer.preload(all_targets)
        self.target_featurizer.cpu()

        if stage == "fit" or stage is None:
            self.data_train = BinaryDataset(
                self.df_train[self._drug_column],
                self.df_train[self._target_column],
                self.df_train[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

            self.data_val = BinaryDataset(
                self.df_val[self._drug_column],
                self.df_val[self._target_column],
                self.df_val[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

        if stage == "test" or stage is None:
            self.data_test = BinaryDataset(
                self.df_test[self._drug_column],
                self.df_test[self._target_column],
                self.df_test[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

    def train_dataloader(self):
        return DataLoader(self.data_train, **self._loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.data_val, **self._loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.data_test, **self._loader_kwargs)
