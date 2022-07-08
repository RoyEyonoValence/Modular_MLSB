import os
import wandb
import torch
import numpy as np
from loguru import logger
from argparse import Namespace
from sklearn import metrics
from sklearn.utils.multiclass import type_of_target
from torch.utils.data import DataLoader
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR, CosineAnnealingWarmRestarts
from modti.utils import get_optimizer, to_numpy


class BaseTrainer(LightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], (dict, Namespace)):
            hparams = args[0]
        elif len(args) == 0 and len(kwargs) > 0:
            hparams = kwargs.pop('hparams', kwargs)
        else:
            raise Exception("Expection a dict or Namespace as inputs")

        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)

        if isinstance(hparams, Namespace):
            self.batch_size = hparams.batch_size if hasattr(hparams, 'batch_size') else 32

        self._model_params = vars(hparams).copy()

        self.lr = self._model_params.pop('lr', 1e-3)
        self.opt = self._model_params.pop('optimizer', 'Adam')
        self.lr_scheduler = self._model_params.pop('lr_scheduler', None)
        self.n_epochs = self._model_params.pop('n_epochs', 100)
        self.precision = self._model_params.pop('precision', 32)
        self.weight_decay = self._model_params.pop('weight_decay', 0)
        self.valid_size = self._model_params.pop('valid_size', 0.2)
        self.hparams.batch_size = self._model_params.pop('batch_size', 32)
        self.early_stopping = self._model_params.pop('early_stopping', False)
        self.auto_scale_batch_size = self._model_params.pop('auto_scale_batch_size', None)
        self.accumulate_grad_batches = self._model_params.pop('accumulate_grad_batches', 1)
        self.amp_backend = self._model_params.pop('amp_backend', 'native')
        self.amp_level = self._model_params.pop('amp_level', '02')
        self.auto_lr_find = self._model_params.pop('auto_lr_find', False)
        self.min_batch_size = self._model_params.pop('min_batch_size', 32)
        self.max_batch_size = self._model_params.pop('max_batch_size', 2048)
        self.min_lr = self._model_params.pop('min_lr', 1e-6)
        self.max_lr = self._model_params.pop('max_lr', 1)
        self.fitted = False
        self.network = None

    def forward(self, *args, **kwargs):
        return self.network(*args, **kwargs)

    def configure_optimizers(self):
        if hasattr(self.network, 'configure_optimizers'):
            return self.network.configure_optimizers()

        opt = get_optimizer(self.opt, filter(lambda p: p.requires_grad, self.network.parameters()),
                            lr=self.lr, weight_decay=self.weight_decay)
        if self.lr_scheduler is None:
            scheduler = None
        elif self.lr_scheduler == "on_plateau":
            scheduler = ReduceLROnPlateau(opt, patience=3, factor=0.5, min_lr=self.lr / 1000)
        elif self.lr_scheduler == "cyclic":
            scheduler = CyclicLR(opt, base_lr=self.lr, max_lr=self.max_lr, cycle_momentum=False)
        elif self.lr_scheduler == "cos_w_restart":
            scheduler = CosineAnnealingWarmRestarts(opt, T_0=8000, eta_min=self.lr/10000)
        else:
            raise Exception("Unexpected lr_scheduler")
        return {'optimizer': opt, 'scheduler': scheduler, "monitor": "train_loss"}

    def train_val_step(self, batch, optimizer_idx=0, train=True, batch_idx=0):
        if hasattr(self.network, 'train_val_step'):
            return self.network.train_val_step(batch, optimizer_idx)
        if len(batch) == 2:
            xs, ys = batch
        else:
            raise Exception("Was expecting a list or tuple of 2 or 3 elements")

        ys_pred = self.network(xs)
        loss_metrics = self.network.compute_loss_metrics(ys_pred, ys)
        prefix = 'train_' if train else 'val_'
        for key, value in loss_metrics.items():
            self.log(prefix+key, value, prog_bar=True, batch_size=len(batch))

        return loss_metrics.get('loss')

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        res = self.train_val_step(batch, train=True, batch_idx=batch_idx)
        result = {'loss': res}
        return result

    def validation_step(self, batch, *args, **kwargs):
        result = self.train_val_step(batch, train=False)
        self.log('checkpoint_on', result, batch_size=len(batch))
        self.log('early_stop_on', result, batch_size=len(batch))
        result = dict(checkpoint_on=result, early_stop_on=result)
        return result

    def train_dataloader(self):
        bs = self.batch_size
        return DataLoader(self._train_dataset, batch_size=bs, shuffle=True,
                          collate_fn=self._train_dataset.collate_fn, num_workers=4)

    def val_dataloader(self):
        bs = self.batch_size
        return DataLoader(self._valid_dataset, batch_size=bs, shuffle=True,
                          collate_fn=self._valid_dataset.collate_fn, num_workers=4)

    def fit(self, train_dataset=None, valid_dataset=None, artifact_dir=None, **kwargs):
        self._train_dataset, self._valid_dataset = train_dataset, valid_dataset

        def get_trainer():
            callbacks = [EarlyStopping(patience=10)] if self.early_stopping else []
            if artifact_dir is not None:
                checkpoint_callback = ModelCheckpoint(filename='{epoch}--{val_loss:.2f}', monitor="checkpoint_on",
                                                      dirpath=os.path.join(artifact_dir, 'checkpoints'),
                                                      verbose=True, mode='min', save_top_k=kwargs['nb_ckpts'],
                                                      save_last=False, every_n_epochs=1)
                callbacks.append(checkpoint_callback)
                callbacks.extend(kwargs.get("extra_callbacks", []))

            res = Trainer(gpus=(1 if torch.cuda.is_available() else 0),
                          max_epochs=self.n_epochs,
                          logger=kwargs.get("loggers", True),
                          default_root_dir=artifact_dir,
                          progress_bar_refresh_rate=int(kwargs['verbose'] > 0),
                          accumulate_grad_batches=self.accumulate_grad_batches,
                          callbacks=callbacks,
                          auto_scale_batch_size=self.auto_scale_batch_size,
                          auto_lr_find=self.auto_lr_find,
                          amp_backend=self.amp_backend,
                          amp_level=self.amp_level,
                          precision=(self.precision if torch.cuda.is_available() else 32),
                          )

            if "initial_weights" in kwargs and kwargs["initial_weights"] is not None:
                logger.info(f"Trying to load a previous checkpoint from {kwargs['initial_weights']}")
                try:
                    self.load_from_checkpoint(kwargs['initial_weights'], strict=True)
                    logger.info("Successfully loaded weights")
                except TypeError as exception:
                    logger.warning(f"Loading the checkpoint failed: {exception}")
            return res

        trainer = get_trainer()
        tuner = Tuner(trainer)

        if (self.auto_scale_batch_size is not None) and self.auto_scale_batch_size:
            logger.info(f"Auto-scaling the batch size in the range [{self.min_batch_size}, {self.max_batch_size}]")
            self.hparams.batch_size = tuner.scale_batch_size(self, steps_per_trial=5, init_val=self.min_batch_size,
                                                             max_trials=int(np.log2(self.max_batch_size/self.min_batch_size)))

        if self.hparams.get('auto_lr_find', False):
            lr_finder_res = tuner.lr_find(self, min_lr=self.hparams.get('min_lr', 1e-6),
                                          max_lr=self.hparams.get('max_lr', 1e-1),
                                          num_training=50, early_stop_threshold=None)
            print(lr_finder_res.results)

        trainer = get_trainer()
        trainer.fit(self)

        self.fitted = True
        return self

    def predict(self, dataset=None):
        ploader = DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=32, num_workers=4)
        res = [to_numpy(self.network.predict(x[0])) for x in ploader]
        return np.concatenate(res, axis=0)

    def evaluate(self, dataset=None):
        ploader = DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=32, num_workers=4)
        preds, targets = zip([(to_numpy(self.network.predict(x[0])), x[1]) for x in ploader])
        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)
        rgr = "continuous" in type_of_target(targets)
        res = {}
        if rgr:
            estimator_metrics = vars(metrics._regression)
        else:
            estimator_metrics = vars(metrics._classification)
        for metric in estimator_metrics:
            try:
                res[metric] = estimator_metrics[metric](targets, preds)
            except:
                res[metric] = None

        return res







