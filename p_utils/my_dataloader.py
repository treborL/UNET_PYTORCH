import logging
import numpy as np
import torch
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import albumentations as A
import cv2 as cv
import os
import random

train_transforms = A.Compose([
    A.PixelDropout(dropout_prob=0.02, drop_value=255),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.GaussNoise(var_limit=255, p=0.25),
    A.CoarseDropout(min_holes=8, max_holes=16, max_height=20,
                    max_width=32, min_height=10, min_width=16, mask_fill_value=0),
    A.ColorJitter()
])


def load_image(filename):
    return cv.imread(str(filename))


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(Path(mask_dir).glob(idx + mask_suffix + '.*'))[0]
    mask = load_image(mask_file)
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


def random_split(items, size):
    sample = set(random.sample(items, size))
    return sorted(sample), sorted(set(items) - sample)


class CustomDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, file_names: list = None, mask_suffix: str = '_mask',
                 mask_values: list = None, is_train_set: bool = False):
        if file_names is None:
            raise RuntimeError('Ids list is None')
        if mask_values is None:
            raise RuntimeError('Mask values list is None')

        self.file_names = file_names
        self.is_train_set = is_train_set

        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.mask_suffix = mask_suffix
        self.mask_values = mask_values

        logging.info(f'Creating dataset with {len(self.file_names)} examples')

    def __len__(self):
        return len(self.file_names)

    @staticmethod
    def preprocess(mask_values, np_img, is_mask):
        w = np_img.shape[1]
        h = np_img.shape[0]
        img = np_img

        if is_mask:
            mask = np.zeros((h, w), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.file_names[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'

        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        if self.is_train_set:
            transformed = train_transforms(image=img, mask=mask)

            img = transformed['image']
            mask = transformed['mask']

        img = self.preprocess(self.mask_values, img, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


def create_train_val_datasets(img_dir: str, mask_dir: str, val_percent: int = 10, mask_suf: str = ''):
    file_names = []
    for file in os.listdir(img_dir):
        file_name = os.fsdecode(file)
        file_name = file_name.split(".png")[0]
        file_names.append(file_name)

    with Pool() as p:
        unique = list(tqdm(
            p.imap(partial(unique_mask_values, mask_dir=mask_dir, mask_suffix=mask_suf),
                   file_names), total=len(file_names)))

    mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))

    validation_count = int(len(file_names) * (val_percent / 100))
    train_count = len(file_names) - validation_count

    train_names, validation_names = random_split(file_names, train_count)

    return CustomDataset(img_dir, mask_dir, train_names, mask_suf, is_train_set=True, mask_values=mask_values), \
        CustomDataset(img_dir, mask_dir, validation_names, mask_suf, is_train_set=False, mask_values=mask_values)
