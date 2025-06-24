import os
import torch
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import zscore
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from monai.transforms import Resize, Compose, RandFlip, RandGaussianNoise, RandAffine
import pytorch_lightning as pl
from collections import Counter

class MRIDataset(Dataset):
    """
    Loads full 3D MRI scans using a .nii cache file, applies z-score normalization and resizing.
    Applies light data augmentation for training.

    """
    def __init__(self, meta_df: pd.DataFrame, data_dir: str, cache_path: str, target_shape=(128, 128, 128), training: bool = False):
        self.meta = meta_df.reset_index(drop=True)
        self.data_dir = data_dir
        self.target_shape = target_shape
        self.training = training
        self.resize = Resize(spatial_size=target_shape)

        # Load the cache file
        with open(cache_path, 'rb') as f:
            self.nii_cache = pickle.load(f)

        # Define transformations
        self.augment = None
        if training:
            self.augment = Compose([
                RandFlip(spatial_axis=[0], prob=0.5),
                RandGaussianNoise(prob=0.2, mean=0.0, std=0.1),
                RandAffine(
                    prob=0.4,
                    rotate_range=(0.1, 0.05, 0.05),
                    scale_range=(0.05, 0.05, 0.05),
                    translate_range=(4, 4, 4),
                    padding_mode="border"
                ),
            ])

    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        nii_path = row["NII_PATH"]

        vol = nib.load(nii_path).get_fdata().astype(np.float32)
        vol = zscore(vol, axis=None).astype(np.float32)

        vol_tensor = torch.from_numpy(vol)[None, ...] # Add channel dimension (1, D, H, W)
        vol_tensor = self.resize(vol_tensor)

        if self.augment:
            vol_tensor = self.augment(vol_tensor)

        label = int(row["Diagnosis_Code"]) # AD= 1, CN = 0
        return vol_tensor, label
    
class MRIDataModule(pl.LightningDataModule):
    """
    Pytorch Lightning DataModule for loading MRI data from full 3D scans.
    """
    def __init__(
            self,
            meta_csv: str,
            data_dir: str,
            cache_path: str,
            batch_size: int = 2,
            num_workers: int = 4,
            target_shape: tuple = (128, 128, 128),
            val_split: float = 0.2,
            test_split: float = 0.1,
    ):
        super().__init__()
        self.meta_csv = meta_csv
        self.data_dir = data_dir
        self.cache_path = cache_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_shape = target_shape
        self.val_split = val_split
        self.test_split = test_split

    def setup(self, stage=None):
        full_df = pd.read_csv(
            self.meta_csv,
            usecols=["RID", "PTID", "EXAMDATE", "Diagnosis_Code"],
            dtype={"RID": str, "Diagnosis_Code": int}
        )
        full_df = full_df[full_df["Diagnosis_Code"].isin([0, 2])].copy()
        full_df["Diagnosis_Code"] = full_df["Diagnosis_Code"].map({0: 0, 2: 1})  # CN=0, AD=1
        full_df["EXAMDATE"] = pd.to_datetime(full_df["EXAMDATE"])
        latest_labels = full_df.sort_values("EXAMDATE").groupby("PTID").tail(1)

        with open(self.cache_path, 'rb') as f:
            nii_cache = pickle.load(f)
        
        matched_rows = []
        for _, row in latest_labels.iterrows():
            ptid_path = os.path.join(self.data_dir, row["PTID"])
            exam_date = row["EXAMDATE"].strftime("%Y-%m-%d")
            cache_key = (ptid_path, exam_date)
            nii_path = nii_cache.get(cache_key)
            if nii_path:
                row = row.copy()
                row["NII_PATH"] = nii_path
                row["EXAMDATE"] = exam_date
                matched_rows.append(row)
            
        df = pd.DataFrame(matched_rows)

        from sklearn.model_selection import train_test_split
        train_df, testval_df = train_test_split(
            df,
            test_size=self.val_split + self.test_split,
            stratify=df["Diagnosis_Code"],
            random_state=42
        )
        val_df, test_df = train_test_split(
            testval_df,
            test_size=self.test_split / (self.val_split + self.test_split),
            stratify=testval_df["Diagnosis_Code"],
            random_state=42
        )

        self.train_ds = MRIDataset(
            train_df,
            self.data_dir,
            self.cache_path,
            target_shape=self.target_shape,
            training=True
        )

        self.val_ds = MRIDataset(
            val_df,
            self.data_dir,
            self.cache_path,
            target_shape=self.target_shape,
            training=False
        )

        self.test_ds = MRIDataset(
            test_df,
            self.data_dir,
            self.cache_path,
            target_shape=self.target_shape,
            training=False
        )

        label_counts = Counter([label for _, label in self.train_ds])
        class_weights = {cls: 1.0 / count for cls, count in label_counts.items()}
        sample_weights = [class_weights[label] for _, label in self.train_ds]
        self.train_sampler = WeightedRandomSampler(
            sample_weights,
            len(sample_weights),
            replacement=True
        )

        print("Final split sizes:")
        print(f" Train: {len(self.train_ds)} | Val: {len(self.val_ds)} | Test: {len(self.test_ds)}")
        print("Train labels:", label_counts)
        print("Val labels:", Counter([y for _, y in self.val_ds]))
        print("Test labels:", Counter([y for _, y in self.test_ds]))

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.train_sampler
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )