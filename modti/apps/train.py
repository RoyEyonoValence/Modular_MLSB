import os
import json
import uuid
import click
import torch
import fsspec
import wandb
import numpy as np

from datetime import datetime
from loguru import logger
from pytorch_lightning.loggers import WandbLogger

from .utils import load_hp, parse_overrides, nested_dict_update
from modti.data import get_dataset, train_val_test_split
from modti.models import get_model


@click.command("train")
@click.argument("config-path", type=click.Path(exists=True))
@click.argument("overrides", nargs=-1, type=str)
@click.option("--wandb-project", default=None, help="If not None, logs the experiment to this WandB project")
def train_cli(config_path, overrides, wandb_project):
    """Train a DTI model

    Loads the configuration as specified by the YAML file at CONFIG_PATH. For quick experimentation, changes to these
    configurations can be specified using the OVERRIDES argument. This uses a specific override syntax:
    For example: `a.b.c=z` will create {'a': {'b': {'c': 'z'}}}
    """
    config = load_hp(conf_path=config_path)
    overrides = parse_overrides(overrides)
    config = nested_dict_update(config, overrides)

    seed = config.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = get_dataset(**config.get("datasets"))
    train, valid, test = train_val_test_split(dataset, val_size=0.2, test_size=0.2)
    logger.info("Succesfully initialized and split the datasets")

    model = get_model(**config.get("model"), **dataset.get_embedding_sizes())
    logger.info("Succesfully initialized model")

    if wandb_project is not None:
        wandb.init(wandb_project, config=config)

    identifier = wandb.run.name if wandb_project is not None else str(uuid.uuid4()).split("-")[0]
    out_directory = os.path.join(config.get("out_directory"), datetime.now().strftime("%Y%m%d"), identifier)
    config["remote_path"] = out_directory

    logger.info('>>> Training configuration : ')
    logger.info(json.dumps(config, sort_keys=True, indent=2))
    logger.info("Training")

    fit_params = config.get("fit_params", {})
    fit_params.update(output_path=out_directory, artifact_dir=out_directory)

    with fsspec.open(os.path.join(out_directory, 'config.json'), 'w') as fd:
        json.dump(config, fd, sort_keys=True, indent=2)

    loggers = True if wandb_project is None else [WandbLogger(log_model=True)]
    model.fit(train_dataset=train, valid_dataset=valid, loggers=loggers, **fit_params)


if __name__ == "__main__":
    train_cli()