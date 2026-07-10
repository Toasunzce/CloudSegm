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
    Cloudsen12 dataset wrapper.
    Used to manage saved sentinel-2 satelline 
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
        returns batch of scenes in [B, 7, 512, 512]
        """
        return torch.stack([self[i][0] for i in range(len(self))])
    
    def getdates(self):
        """
        returns list of scene dates
        """
        return [self[i][2] for i in range(len(self))]