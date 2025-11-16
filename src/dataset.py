import os
import random
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
from omegaconf import DictConfig
from PIL import Image

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def pad_to_square(img, fill):
    w, h = img.size
    if w == h:
        return img
    max_side = max(w, h)
    pad_w = max_side - w
    pad_h = max_side - h
    padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
    return TF.pad(img, padding, fill=fill)


class JointTransform:
    def __init__(self, resize: Tuple[int, int], train: bool):
        self.resize = resize
        self.train = train

    def __call__(self, img, mask):
        img = pad_to_square(img, fill=0)
        mask = pad_to_square(mask, fill=2)

        if self.train:
            if random.random() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)

            angle = random.uniform(-10, 10)
            translations = (
                random.uniform(-0.1, 0.1) * img.size[0],
                random.uniform(-0.1, 0.1) * img.size[1],
            )
            scale = random.uniform(0.9, 1.1)
            img = TF.affine(
                img,
                angle=angle,
                translate=translations,
                scale=scale,
                shear=0,
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
            mask = TF.affine(
                mask,
                angle=angle,
                translate=translations,
                scale=scale,
                shear=0,
                interpolation=InterpolationMode.NEAREST,
                fill=1,
            )

        img = TF.resize(img, self.resize, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.resize, interpolation=InterpolationMode.NEAREST_EXACT)

        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=mean, std=std)
        mask = torch.as_tensor(np.array(mask, dtype=np.int64) - 1)
        return img, mask


class OxfordPetSegmentation(Dataset):
    def __init__(self, cfg: DictConfig, split: str, train: bool):
        self.base = OxfordIIITPet(
            root=cfg.data.root,
            split=split,
            target_types="segmentation",
            download=True,
        )
        self.transform = JointTransform(tuple(cfg.data.resize), train=train)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, mask = self.base[idx]
        return self.transform(img, mask)


class InferenceDataset(Dataset):
    def __init__(self, folder_path, resize):
        self.paths = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith("png")
        ]
        self.resize = tuple(resize)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        raw = pad_to_square(img, fill=0)
        raw = TF.resize(raw, self.resize, interpolation=InterpolationMode.BILINEAR)
        raw_display = raw.copy()
        tensor = TF.to_tensor(raw)
        tensor = TF.normalize(tensor, mean=mean, std=std)
        return tensor, raw_display


def data_loaders(cfg: DictConfig):
    train_ds = OxfordPetSegmentation(cfg, split="trainval", train=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=True)

    test_ds = OxfordPetSegmentation(cfg, split="test", train=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.data.batch_size, shuffle=False)

    inference_ds = InferenceDataset(os.path.join(cfg.data.root, "Inference"), cfg.data.resize)
    inference_loader = DataLoader(inference_ds, batch_size=1, shuffle=False, collate_fn=lambda batch: batch[0])
    return train_loader, test_loader, inference_loader
