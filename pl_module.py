# base
import torch
import pytorch_lightning as pl
# type hint
from typing import Dict
from torch.nn import Module
# optim
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
# model
from model.shallow_arch import UWnet
# loss
from loss.perceptual_loss import PerceptualLoss
from loss.charbonnier_loss import CharbonnierLoss
# metric
from kornia.metrics import ssim, psnr
# log
from torchvision.utils import make_grid


class DenosingModule(pl.LightningModule):
    def __init__(self, hparams: Dict):
        super(DenosingModule, self).__init__()
        self.save_hyperparameters(hparams)
        self._model: Module = UWnet()
        self._loss_char: Module = CharbonnierLoss()
        self._loss_perc: Module = PerceptualLoss()

    def forward(self, x):
        return self._model(x)

    def configure_optimizers(self):
        optimizer = AdamW(
            params=self._model.parameters(),
            lr=self.hparams.optim['lr_init'],
            weight_decay=self.hparams.optim['weight_decay']
        )
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.hparams.train['epochs'] + 5,
            eta_min=self.hparams.optim['lr_min']
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        input_img, target_img = batch
        output_img = self._model(input_img)
        output_img = torch.clamp(output_img, 0, 1)
        train_loss = (self._loss_char(output_img, target_img)
                      + 0.02 * self._loss_perc(output_img, target_img))
        self.log('train_loss', train_loss)
        return {'loss': train_loss}

    def validation_step(self, batch, batch_idx):
        input_img, target_img = batch
        output_img = self._model(input_img)
        valid_loss = (self._loss_char(output_img, target_img)
                      + 0.02 * self._loss_perc(output_img, target_img))
        valid_ssim = ssim(output_img, target_img, 5).mean().item()
        valid_psnr = psnr(output_img, target_img, 1).item()
        self.log('psnr', valid_psnr, on_step=False, on_epoch=True)
        self.log('ssim', valid_ssim, on_step=False, on_epoch=True)
        self.log('valid_loss', valid_loss, on_step=False, on_epoch=True)
        if batch_idx % 10 == 0:
            input_img = make_grid(input_img)
            output_img = make_grid(output_img)
            target_img = make_grid(target_img)
            self.logger.experiment.add_image('input_img', input_img)
            self.logger.experiment.add_image('output_img', output_img)
            self.logger.experiment.add_image('target_img', target_img)
        return {'valid_loss': valid_loss, 'psnr': valid_psnr, 'ssim': valid_ssim}

