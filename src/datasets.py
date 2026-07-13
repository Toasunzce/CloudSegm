"""
Dataset classes for cloud segmentation pipeline.

CloudDataset - Cloudsen12 patches for model training and validation.
CustomSceneDataset - Sentinel-2 scense fetched via src.fetching.fetch_and_save_patches() for inference.
"""



import os
import pandas as pd
import numpy as np
import rasterio as rio
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision import tv_tensors



class CloudDataset(Dataset):
    """
    CloudSen12 dataset wrapper for model training and validation.

    Expects a directory with the following structure:
        data_dir/
            metadata.csv
            train/
                000001_img.tif
                000001_label.tif
                ...
            val/
            test/

    Images are 7-band Sentinel-2 L1C patches (B02 B03 B04 B08 B10 B11 B12),
    stored as uint16 GeoTIFF and normalized to [0, 1].

    Masks contain 4 classes:
        0 - clear
        1 - thick cloud
        2 - thin cloud
        3 - cloud shadow
    """
    def __init__(self, data_dir, split):
        meta = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
        self.meta = meta[meta["split"] == split].reset_index(drop=True)
        self.data_dir = data_dir
        self.split = split

        self.tf = v2.Compose([
            v2.RandomHorizontalFlip(0.5),
            v2.RandomVerticalFlip(0.5),
            v2.RandomResizedCrop(size=512, scale=(0.6, 1.0), antialias=True), 
            v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.3),
        ]) if split == "train" else None

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):  # ty:ignore[invalid-method-override]
        row = self.meta.iloc[i]

        with rio.open(os.path.join(self.data_dir, row["img"])) as src:
            img = src.read().astype(np.float32) / 3000.0
        with rio.open(os.path.join(self.data_dir, row["label"])) as src:
            mask = src.read(1).astype(np.int64)

        x = torch.from_numpy(img)                          # float32 (7, H, W)
        y = tv_tensors.Mask(torch.from_numpy(mask))           # int64  (H, W)

        if self.tf is not None:
            x, y = self.tf(x, y)

        return x, y.long()


class CustomSceneDataset(Dataset):
    """
    Dataset for Sentinel-2 scenes fetched via CDSE API (see fetching.py).

    Expects a directory produced by fetch_and_save_patches():
        location_dir/
            metadata.csv
            minsk_20240701_img.tif
            minsk_20240706_img.tif
            ...
    
    Unlike CloudDataset, this class has no augmentations and returns
    scene metadata (s2_id, date)
    """
    def __init__(self, location_dir):
        self.location_dir = location_dir
        self.meta = pd.read_csv(os.path.join(location_dir, "metadata.csv"))

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        row = self.meta.iloc[index]
        with rio.open(os.path.join(self.location_dir, row["img"])) as src:
            arr = src.read().astype(np.float32)
        
        arr = np.clip(arr - 1000.0, 0, None)
        img = arr / 10000.0

        return torch.from_numpy(img), row["s2_id"], row["date"]
    
    def getscenes(self):
        """
        Load all scenes into a batch.
        Returns: (N, 7, 512, 512) float32
        """
        return torch.stack([self[i][0] for i in range(len(self))])
    
    def getdates(self):
        """
        Return acquisition dates for all scenes.
        """
        return [self[i][2] for i in range(len(self))]