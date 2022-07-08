import copy
import click
import torch
import wandb
import typing as T
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from omegaconf import OmegaConf
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    roc_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    auc,
    precision_recall_curve,
)
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from modti.models import layers as dti_architecture


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


def get_config(mol_feat, prot_feat):
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
    }

    return OmegaConf.structured(cfg)


def get_model(model_type, **model_kwargs):
    try:
        return getattr(dti_architecture, model_type)(**model_kwargs)
    except AttributeError:
        raise ValueError("Specified model is not supported")


def flatten(d):
    d_ = {}
    if not isinstance(d, T.Mapping):
        return d
    for k, v in d.items():
        if isinstance(v, T.Mapping):
            d_flat = flatten(v)
            for k_, v_ in d_flat.items():
                d_[k_] = v_
        else:
            d_[k] = v
    return d_


def get_task(task_name):
    if task_name.lower() == "biosnap":
        return "./datasets/BIOSNAP/full_data"
    elif task_name.lower() == "bindingdb":
        return "./datasets/BindingDB"
    elif task_name.lower() == "davis":
        return "./datasets/DAVIS"
    elif task_name.lower() == "biosnap_prot":
        return "./datasets/BIOSNAP/unseen_protein"
    elif task_name.lower() == "biosnap_mol":
        return "./datasets/BIOSNAP/unseen_drug"


def test(data_generator, model):
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0
    for i, (d, p, label) in tqdm(
        enumerate(data_generator), total=len(data_generator)
    ):
        score = model(d.cuda(), p.cuda())

        m = torch.nn.Sigmoid()
        logits = torch.squeeze(m(score))
        loss_fct = torch.nn.BCELoss()

        label = Variable(torch.from_numpy(np.array(label)).float()).cuda()

        loss = loss_fct(logits, label)

        loss_accumulate += loss
        count += 1

        logits = logits.detach().cpu().numpy()

        label_ids = label.to("cpu").numpy()
        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + logits.flatten().tolist()

    loss = loss_accumulate / count

    fpr, tpr, thresholds = roc_curve(y_label, y_pred)

    precision, recall, thresholds = precision_recall_curve(y_label, y_pred)

    all_f1_scores = []
    for t in thresholds:
        all_f1_scores.append(f1_score(y_label, (y_pred >= t).astype(int)))

    thred_optim = thresholds[np.argmax(all_f1_scores)]

    print("optimal threshold: " + str(thred_optim))

    y_pred_s = (y_pred >= thred_optim).astype(int)

    auc_k = auc(fpr, tpr)
    print("AUROC:" + str(auc_k))
    print("AUPRC: " + str(average_precision_score(y_label, y_pred)))

    cm1 = confusion_matrix(y_label, y_pred_s)
    print("Confusion Matrix : \n", cm1)
    print("Recall : ", recall_score(y_label, y_pred_s))
    print("Precision : ", precision_score(y_label, y_pred_s))

    total1 = sum(sum(cm1))
    # from confusion matrix calculate accuracy
    accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
    print("Accuracy : ", accuracy1)

    sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    print("Sensitivity : ", sensitivity1)

    specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    print("Specificity : ", specificity1)

    return (
        roc_auc_score(y_label, y_pred),
        average_precision_score(y_label, y_pred),
        f1_score(y_label, y_pred_s),
        accuracy1,
        sensitivity1,
        specificity1,
        y_pred,
        loss.item(),
    )

@click.command(description="PLM_DTI Training.")
@click.argument("datasets", type=str, choices=["biosnap", "bindingdb", "davis", "biosnap_prot", "biosnap_mol"],
    help="Dataset name. Could be biosnap, bindingdb, davis, biosnap_prot, biosnap_mol.")
@click.argument("mol-featurizer", help="Molecule featurizer", dest="mol_feat")
@click.argument("prot-featurizer", help="Molecule featurizer", dest="prot_feat")
@click.option("arch-type", type=str, default="SimpleCosine", help="Model architecture")

@click.option("batch-size", default=16, type=int,
              help="Batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel")
@click.option("latent-dist", default="Cosine", type=str, help="Distance in embedding space to supervise with",    dest="latent_dist",)
@click.option("nb-epochs", default=50, type=int, help="number of total epochs to run",)
@click.option("lr", default=1e-4, type=float, help="initial learning rate (default: 1e-4)")
@click.option("seed", default=0, type=int, help="Seed")
@click.option("checkpoint", default=None, help="Model weights to start from")
@click.option("wandb-project", default=None, help="If not None, logs the experiment to this WandB project")
def main(dataset, mol_featurizer, prot_featurizer, arch_type, batch_size, latent_dist,
         nb_epochs, lr, seed, device, checkpoint, wandb_project):

    torch.manual_seed(seed)  # reproducible torch:2 np:3
    np.random.seed(seed)
    config = get_config(mol_featurizer, prot_featurizer)
    config.data.pool = config.model.arch_type != "LSTMCosine"

    loss_history = []

    print("--- Data Preparation ---")

    config.data.to_disk_path = f"saved_embeddings/{dataset}"

    dataFolder = get_task(args.task)

    print("--- loading dataframes ---")
    df_train = pd.read_csv(dataFolder + "/train.csv", header=0, index_col=0)
    df_val = pd.read_csv(dataFolder + "/val.csv", header=0, index_col=0)
    df_test = pd.read_csv(dataFolder + "/test.csv", header=0, index_col=0)

    print("--- loading dataloaders ---")
    (
        training_generator,
        validation_generator,
        testing_generator,
        mol_emb_size,
        prot_emb_size,
    ) = get_dataloaders(df_train, df_val, df_test, **config.data)

    config.model.mol_emb_size, config.model.prot_emb_size = (
        mol_emb_size,
        prot_emb_size,
    )
    config.model.distance_metric = args.latent_dist

    print("--- getting model ---")
    if args.checkpoint is None:
        model = plm_dti.get_model(**config.model)
    else:
        model = torch.load(args.checkpoint)
    model = model.cuda()
    opt = torch.optim.Adam(model.parameters(), lr=config.training.lr)

    print("--- loading wandb ---")
    wandb.init(
        project=args.wandb_proj,
        name=config.experiment_id,
        config=flatten(config),
    )
    wandb.watch(model, log_freq=100)

    # early stopping
    max_auprc = 0
    model_max = copy.deepcopy(model)

    # with torch.set_grad_enabled(False):
    #     auc, auprc, f1, logits, loss = test(testing_generator, model_max)
    #     # wandb.log({"test/loss": loss, "epoch": 0,
    #     #            "test/auc": float(auc),
    #     #            "test/aupr": float(auprc),
    #     #            "test/f1": float(f1),
    #     #           })
    #     print('Initial Testing AUROC: ' + str(auc) + ' , AUPRC: ' + str(auprc) + ' , F1: ' + str(
    #         f1) + ' , Test loss: ' + str(loss))

    print("--- Go for Training ---")
    torch.backends.cudnn.benchmark = True
    tg_len = len(training_generator)
    start_time = time()
    for epo in range(config.training.n_epochs):
        model.train()
        epoch_time_start = time()
        for i, (d, p, label) in enumerate(training_generator):
            score = model(d.cuda(), p.cuda())

            label = Variable(torch.from_numpy(np.array(label)).float()).cuda()

            loss_fct = torch.nn.BCELoss()
            m = torch.nn.Sigmoid()
            n = torch.squeeze(m(score))

            loss = loss_fct(n, label)
            loss_history.append(loss)
            wandb.log(
                {
                    "train/loss": loss,
                    "epoch": epo,
                    "step": epo * tg_len * args.batch_size
                    + i * args.batch_size,
                }
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % 1000 == 0:
                print(
                    "Training at Epoch "
                    + str(epo + 1)
                    + " iteration "
                    + str(i)
                    + " with loss "
                    + str(loss.cpu().detach().numpy())
                )

        epoch_time_end = time()
        if epo % config.training.every_n_val == 0:
            with torch.set_grad_enabled(False):
                (
                    val_auc,
                    val_auprc,
                    val_f1,
                    val_accuracy,
                    val_sensitivity,
                    val_specificity,
                    val_logits,
                    val_loss,
                ) = test(validation_generator, model)
                wandb.log(
                    {
                        "val/loss": val_loss,
                        "epoch": epo,
                        "val/auc": float(val_auc),
                        "val/aupr": float(val_auprc),
                        "val/f1": float(val_f1),
                        "val/acc": float(val_accuracy),
                        "val/sens": float(val_sensitivity),
                        "val/spec": float(val_specificity),
                        "Charts/epoch_time": (
                            epoch_time_end - epoch_time_start
                        )
                        / config.training.every_n_val,
                    }
                )
                if val_auprc > max_auprc:
                    model_max = copy.deepcopy(model)
                    max_auprc = val_auprc
                print(
                    "Validation at Epoch "
                    + str(epo + 1)
                    + " , AUROC: "
                    + str(val_auc)
                    + " , AUPRC: "
                    + str(val_auprc)
                    + " , F1: "
                    + str(val_f1)
                )

    end_time = time()
    print("--- Go for Testing ---")
    try:
        with torch.set_grad_enabled(False):
            model_max = model_max.eval()
            test_start_time = time()
            (
                test_auc,
                test_auprc,
                test_f1,
                test_accuracy,
                test_sensitivity,
                test_specificity,
                test_logits,
                test_loss,
            ) = test(testing_generator, model_max)
            test_end_time = time()
            wandb.log(
                {
                    "test/loss": test_loss,
                    "epoch": epo,
                    "test/auc": float(test_auc),
                    "test/aupr": float(test_auprc),
                    "test/f1": float(test_f1),
                    "test/acc": float(test_accuracy),
                    "test/sens": float(test_sensitivity),
                    "test/spec": float(test_specificity),
                    "test/eval_time": (test_end_time - test_start_time),
                    "Charts/wall_clock_time": (end_time - start_time),
                }
            )
            print(
                "Testing AUROC: "
                + str(test_auc)
                + " , AUPRC: "
                + str(test_auprc)
                + " , F1: "
                + str(test_f1)
                + " , Test loss: "
                + str(test_loss)
            )
            # trained_model_artifact = wandb.Artifact(conf.experiment_id, type="model")
            torch.save(
                model_max, f"best_models/{config.experiment_id}_best_model.sav"
            )
    except Exception as e:
        logg.error(f"testing failed with exception {e}")
    return model_max, loss_history


s = time()
model_max, loss_history = main()
e = time()
print(e - s)
