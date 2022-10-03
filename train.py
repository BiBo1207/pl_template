import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from pl_module import DenosingModule
from data.uieb_data import UIEBDataModule
from pytorch_lightning.callbacks import (EarlyStopping, ModelCheckpoint,
                                         LearningRateMonitor)
from util.train_utils import last_model_path


def load_callbacks():
    callbacks = [
        EarlyStopping(
            monitor='valid_loss',
            mode='min',
            patience=10,
            min_delta=0.001,
            check_on_train_epoch_end=False
        ),
        ModelCheckpoint(
            monitor='psnr',
            filename='{epoch: 02d}_{psnr: .2f}_{ssim: .3f}',
            save_top_k=3,
            mode='max',
            save_last=True
        ),
        LearningRateMonitor(
            logging_interval='epoch'
        )
    ]
    return callbacks


def train(hparam):
    pl.seed_everything(hparam['train']['seed'])
    logger = TensorBoardLogger(save_dir=hparam['train']['save_dir'])
    trainer = pl.Trainer(
        logger=logger,
        callbacks=load_callbacks(),
        max_epochs=hparam['train']['epochs'],
        accelerator=hparam['train']['device'],
        precision=32
    )
    pl_module = DenosingModule(hparam)
    data_module = UIEBDataModule(hparam)
    if hparam['train']['resume']:
        trainer.fit(model=pl_module, datamodule=data_module, ckpt_path=last_model_path(0))
    else:
        trainer.fit(model=pl_module, datamodule=data_module)



