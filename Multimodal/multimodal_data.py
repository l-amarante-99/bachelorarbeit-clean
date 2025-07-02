import os
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from monai.transforms import Compose, Resize, RandFlip, RandGaussianNoise, RandAffine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pytorch_lightning as pl
import nibabel as nib
from collections import Counter

class MultimodalDataset(Dataset):
    def __init__(self, df: DataFrame, clinical_features: list, target_shape=(96,96,96), training=False):
        self.df = df.reset_index(drop=True)
        self.training = training
        self.resize = Resize(spatial_size=target_shape)

        self.augment = Compose([
            RandFlip(spatial_axis=[0], prob=0.5),
            RandAffine(
                rotate_range=(0.1, 0.05, 0.05),
                translate_range=(4, 4, 4),
                scale_range=(0.05, 0.05, 0.05),
                padding_mode="border",
                prob=0.4
            ),
            RandGaussianNoise(prob=0.2, mean=0.0, std=0.1),
        ]) if training else None

        features = df[clinical_features].values.astype(np.float32)
        imputer = SimpleImputer(strategy='mean')
        features_imputed = imputer.fit_transform(features)
        scaler = StandardScaler()
        self.features_scaled = scaler.fit_transform(features_imputed)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vol = nib.load(row['NII_PATH']).get_fdata().astype(np.float32)
        vol = (vol - vol.mean()) / (vol.std() + 1e-8)  # Normalize volume
        vol_tensor = torch.from_numpy(vol)[None, ...] # shape (1, D, H, W)
        vol_tensor = self.resize(vol_tensor)

        if self.augment:
            vol_tensor = self.augment(vol_tensor)

        clinical = torch.tensor(self.features_scaled[idx], dtype=torch.float32)
        label = int(row['Diagnosis_Code'])

        return (vol_tensor, clinical), label
    
class MultimodalDataModule(pl.LightningDataModule):
    def __init__(self, merged_csv_path, data_dir, cache_path, batch_size=4, num_workers=4, val_split=0.2, test_split=0.1, target_shape=(96, 96, 96)):
        super().__init__()
        self.merged_csv_path = merged_csv_path
        self.data_dir = data_dir
        self.cache_path = cache_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.target_shape = target_shape

    def setup(self, stage=None):
        df = pd.read_csv(self.merged_csv_path)
        df['EXAMDATE'] = pd.to_datetime(df['EXAMDATE'])
        df = df.sort_values('EXAMDATE').groupby('PTID').tail(1)

        with open(self.cache_path, 'rb') as f:
            nii_cache = pickle.load(f)

        matched_rows = []
        for _, row in df.iterrows():
            ptid_path = os.path.join(self.data_dir, row['PTID'])
            exam_date = row['EXAMDATE'].strftime('%Y-%m-%d')
            cache_key = (ptid_path, exam_date)
            nii_path = nii_cache.get(cache_key)
            if nii_path:
                row = row.copy()
                row['EXAMDATE'] = exam_date
                row['NII_PATH'] = nii_path
                matched_rows.append(row)
        
        df = pd.DataFrame(matched_rows)
        df = df.dropna(subset=['Diagnosis_Code'])

        clinical_features = [col for col in df.columns if col not in ['PTID', 'EXAMDATE', 'NII_PATH', 'Diagnosis_Code']]
        self.clinical_features = clinical_features

        train_df, testval_df = train_test_split(df, test_size=self.val_split + self.test_split, stratify=df['Diagnosis_Code'], random_state=42)
        val_df, test_df = train_test_split(testval_df, test_size=self.test_split / (self.val_split + self.test_split), stratify=testval_df['Diagnosis_Code'], random_state=42)

        self.train_ds = MultimodalDataset(train_df, clinical_features, self.target_shape, training=True)
        self.val_ds = MultimodalDataset(val_df, clinical_features, self.target_shape, training=False)
        self.test_ds = MultimodalDataset(test_df, clinical_features, self.target_shape, training=False)

        label_counts = Counter([y for _, y in self.train_ds])
        class_weights = {cls: 1.0 / count for cls, count in label_counts.items()}
        sample_weights = [class_weights[label] for _, label in self.train_ds]
        self.train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        print("Final split sizes:")
        print(f"Train: {len(self.train_ds)} | Val: {len(self.val_ds)} | Test: {len(self.test_ds)}")
        print("Train labels:", label_counts)
        print("Val labels:", Counter([y for _, y in self.val_ds]))
        print("Test labels:", Counter([y for _, y in self.test_ds]))

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, sampler=self.train_sampler, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)