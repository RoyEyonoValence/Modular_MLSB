import torch
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence

from .models import layers as dti_architecture
from . import protein as protein_features
from . import molecule as molecule_features

##################
# Data Set Utils #
##################


def molecule_protein_collate_fn(args, pad=False):
    """
    Collate function for PyTorch data loader.

    :param args: Batch of training samples with molecule, protein, and affinity
    :type args: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    """
    memb = [a[0] for a in args]
    pemb = [a[1] for a in args]
    labs = [a[2] for a in args]

    if pad:
        proteins = pad_sequence(pemb, batch_first=True)
    else:
        proteins = torch.stack(pemb, 0)
    molecules = torch.stack(memb, 0)
    affinities = torch.stack(labs, 0)

    return molecules, proteins, affinities


class DTIDataset(Dataset):
    def __init__(self, smiles, sequences, labels, mfeats, pfeats):
        assert len(smiles) == len(sequences)
        assert len(sequences) == len(labels)
        self.smiles = smiles
        self.sequences = sequences
        self.labels = labels

        self.mfeats = mfeats
        self.pfeats = pfeats

    def __len__(self):
        return len(self.smiles)

    @property
    def shape(self):
        return self.mfeats._size, self.pfeats._size

    def __getitem__(self, i):
        memb = self.mfeats(self.smiles[i])
        pemb = self.pfeats(self.sequences[i])
        lab = torch.tensor(self.labels[i])

        return memb, pemb, lab


#################
# API Functions #
#################

def get_dataloaders(
    train_df,
    val_df,
    test_df,
    batch_size,
    shuffle,
    num_workers,
    mol_feat,
    prot_feat,
    pool=True,
    precompute=True,
    to_disk_path=None,
    device=0,
):

    df_values = {}
    all_smiles = []
    all_sequences = []
    for df, set_name in zip(
        [train_df, val_df, test_df], ["train", "val", "test"]
    ):
        all_smiles.extend(df["SMILES"])
        all_sequences.extend(df["Target Sequence"])
        # df_thin = df[["SMILES", "Target Sequence", "Label"]]
        df_values[set_name] = (
            df["SMILES"],
            df["Target Sequence"],
            df["Label"],
        )

    try:
        mol_feats = getattr(molecule_features, mol_feat)()
    except AttributeError:
        raise ValueError(
            f"Specified molecule featurizer {mol_feat} is not supported"
        )
    try:
        prot_feats = getattr(protein_features, prot_feat)(pool=pool)
    except AttributeError:
        raise ValueError(
            f"Specified protein featurizer {prot_feat} is not supported"
        )
    if precompute:
        mol_feats.precompute(
            all_smiles, to_disk_path=to_disk_path, from_disk=True
        )
        prot_feats.precompute(
            all_sequences, to_disk_path=to_disk_path, from_disk=True
        )

    loaders = {}
    for set_name in ["train", "val", "test"]:
        smiles, sequences, labels = df_values[set_name]

        dataset = DTIDataset(smiles, sequences, labels, mol_feats, prot_feats)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=lambda x: molecule_protein_collate_fn(x, pad=not pool),
        )
        loaders[set_name] = dataloader

    return tuple(
        [
            loaders["train"],
            loaders["val"],
            loaders["test"],
            mol_feats._size,
            prot_feats._size,
        ]
    )


def get_config(experiment_id, mol_feat, prot_feat):
    data_cfg = {
        "batch_size": 32,
        "num_workers": 0,
        "precompute": True,
        "mol_feat": mol_feat,
        "prot_feat": prot_feat,
    }
    model_cfg = {
        # "latent_size": 1024,
        # "distance_metric": "Cosine"
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


def get_model(model_type, **model_kwargs):
    try:
        return getattr(dti_architecture, model_type)(**model_kwargs)
    except AttributeError:
        raise ValueError("Specified model is not supported")
