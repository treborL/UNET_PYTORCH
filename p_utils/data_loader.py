import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import albumentations as A
import cv2 as cv

transforms = A.Compose([
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
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = load_image(mask_file)
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class CustomDataloader(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, mask_suffix: str = '_mask'):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')


        #with Pool() as p:
        #    unique = list(tqdm(
        #       p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
        #        total=len(self.ids)
        #    ))

        self.mask_values = [[1, 1, 1], [2, 2, 2]] # list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))[1:]
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, np_img, is_mask, scale=1):
        w = np_img.shape[1]
        h = np_img.shape[0]

        #newW, newH = int(scale * w), int(scale * h)
        #assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        #pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        #img = np.asarray(pil_img)
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
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'

        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        transformed = transforms(image=img, mask=mask)

        img = transformed['image']
        mask = transformed['mask']

        img = self.preprocess(self.mask_values, img, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }
