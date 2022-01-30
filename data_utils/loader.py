"""
__author__: <amangupta0044>
Contact: amangupta0044@gmail.com
Created: Sunday, 14th January 2022
Last-modified Monday, 14th January 2022
"""


from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import os
import cv2
from data_utils.transform import (
    get_preprocessing,
    get_validation_augmentation,
    get_training_augmentation,
)

import segmentation_models_pytorch as smp
import numpy as np


class Dataset(BaseDataset):
    """Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = ["background", "coords"]

    def __init__(
        self,
        images_dir,
        masks_dir,
        classes=None,
        augmentation=None,
        preprocessing=None,
    ):

        self.ids = os.listdir(images_dir)
        assert len(self.ids) != 0
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype("float")

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask

    def __len__(self):
        return len(self.ids)


def get_loader(cfg, split_type: str = "train"):

    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        cfg.model.ENCODER, cfg.model.ENCODER_WEIGHTS
    )
    x_dir, y_dir = (
        os.path.join(cfg.processing.DATA_PATH, split_type),
        os.path.join(cfg.processing.DATA_PATH, f"{split_type}annot"),
    )

    if split_type == "train":

        dataset = Dataset(
            x_dir,
            y_dir,
            augmentation=get_training_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn),
            classes=cfg.model.CLASSES,
        )
        loader = DataLoader(
            dataset,
            batch_size=cfg.processing.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=cfg.processing.TRAIN_WORKERS,
        )
        return loader

    elif split_type == "val" or split_type == "test":

        dataset = Dataset(
            x_dir,
            y_dir,
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn),
            classes=cfg.model.CLASSES,
        )
        loader = DataLoader(
            dataset,
            batch_size=cfg.processing.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.processing.VALID_WORKERS,
        )

        return loader
