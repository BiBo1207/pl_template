import os
import random
from PIL import Image
from typing import Dict
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as ttf
from torchvision.transforms import (RandomCrop, Pad, RandomHorizontalFlip,
                                    RandomVerticalFlip, Resize, ToTensor)
from pytorch_lightning import LightningDataModule


class UIEBTrain(Dataset):
    _INPUT_ = 'input'
    _TARGET_ = 'gt'

    def __init__(self, folder: str, size: int):
        super(UIEBTrain, self).__init__()
        self._size = size
        self._root = folder
        self._filenames = os.listdir(os.path.join(self._root, self._INPUT_))

    def __len__(self):
        return len(self._filenames)

    def __getitem__(self, item):
        input_img = Image.open(os.path.join(self._root, self._INPUT_, self._filenames[item]))
        target_img = Image.open(os.path.join(self._root, self._TARGET_, self._filenames[item]))
        input_img, target_img = self._aug_data(input_img, target_img)
        return input_img, target_img

    def _aug_data(self, input_img, target_img):
        # padding
        pad_w = self._size - input_img.width if input_img.width < self._size else 0
        pad_h = self._size - input_img.height if input_img.height < self._size else 0
        input_img = Pad(padding=(0, 0, pad_w, pad_h), padding_mode='reflect')(input_img)
        target_img = Pad(padding=(0, 0, pad_w, pad_h), padding_mode='reflect')(target_img)
        # random crop
        i, j, h, w = RandomCrop.get_params(input_img, output_size=(self._size, self._size))
        input_img = ttf.crop(input_img, i, j, h, w)
        target_img = ttf.crop(target_img, i, j, h, w)
        # random flip
        rand_flip = random.randint(0, 1)
        input_img = RandomVerticalFlip(rand_flip)(input_img)
        input_img = RandomHorizontalFlip(rand_flip)(input_img)
        target_img = RandomVerticalFlip(rand_flip)(target_img)
        target_img = RandomHorizontalFlip(rand_flip)(target_img)
        # random rotate
        rand_rotate = random.randint(0, 3)
        input_img = ttf.rotate(input_img, 90 * rand_rotate)
        target_img = ttf.rotate(target_img, 90 * rand_rotate)
        # to tensor
        input_img = ToTensor()(input_img)
        target_img = ToTensor()(target_img)
        return input_img, target_img


class UIEBValid(Dataset):
    _INPUT_ = 'input'
    _TARGET_ = 'gt'

    def __init__(self, folder: str, size: int):
        super(UIEBValid, self).__init__()
        self._size = size
        self._root = folder
        self._filenames = os.listdir(os.path.join(self._root, self._INPUT_))

    def __len__(self):
        return len(self._filenames)

    def __getitem__(self, item):
        input_img = Image.open(os.path.join(self._root, self._INPUT_, self._filenames[item]))
        target_img = Image.open(os.path.join(self._root, self._TARGET_, self._filenames[item]))
        input_img, target_img = self._aug_data(input_img, target_img)
        return input_img, target_img

    def _aug_data(self, input_img, target_img):
        # padding
        pad_w = self._size - input_img.width if input_img.width < self._size else 0
        pad_h = self._size - input_img.height if input_img.height < self._size else 0
        input_img = Pad(padding=(0, 0, pad_w, pad_h), padding_mode='reflect')(input_img)
        target_img = Pad(padding=(0, 0, pad_w, pad_h), padding_mode='reflect')(target_img)
        # random crop
        i, j, h, w = RandomCrop.get_params(input_img, output_size=(self._size, self._size))
        input_img = ttf.crop(input_img, i, j, h, w)
        target_img = ttf.crop(target_img, i, j, h, w)
        # to tensor
        input_img = ToTensor()(input_img)
        target_img = ToTensor()(target_img)
        return input_img, target_img


class UIEBDataModule(LightningDataModule):
    def __init__(self, hparam: Dict):
        super(UIEBDataModule, self).__init__()
        self.save_hyperparameters(hparam)
        pass

    def train_dataloader(self):
        train_data = UIEBTrain(
            folder=self.hparams.data['train']['dir'],
            size=self.hparams.data['train']['img_size']
        )
        return DataLoader(
            dataset=train_data,
            batch_size=self.hparams.data['train']['batch_size'],
            num_workers=self.hparams.data['num_works'],
            pin_memory=self.hparams.data['pin_memory'],
            shuffle=True
        )

    def val_dataloader(self):
        valid_data = UIEBValid(
            folder=self.hparams.data['valid']['dir'],
            size=self.hparams.data['valid']['img_size']
        )
        return DataLoader(
            dataset=valid_data,
            batch_size=self.hparams.data['valid']['batch_size'],
            num_workers=self.hparams.data['num_works'],
            pin_memory=self.hparams.data['pin_memory'],
        )
