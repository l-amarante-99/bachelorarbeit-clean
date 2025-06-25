import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from collections import Counter

class ClinicalDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        # Keep only CN (0) and AD (2) labels, then relabel: CN = 0, AD = 1
        df = df[df["label"].isin([0,2])]
        df["label"] = df["label"].replace({2: 1})

        df = df.drop(columns=["PTID", "RID"])

        # One-hot encode Genotype
        genotype_dummies = pd.get_dummies(df["GENOTYPE"], prefix="GENO")
        df = df.drop(columns=["GENOTYPE"])
        df = pd.concat([df, genotype_dummies], axis=1)

        labels = df["label"]
        features = df.drop(columns=["label"])

        # Impute missing values with mean
        imputer = SimpleImputer(strategy='mean')
        features_imputed = imputer.fit_transform(features)

        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_imputed)

        self.X = torch.tensor(features_scaled, dtype=torch.float32)
        self.y = torch.tensor(labels.values, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx] 
    

class ClinicalDataModule(pl.LightningDataModule):
    def __init__(self, csv_path, batch_size=64, val_split=0.2, test_split=0.1):
        super().__init__()
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split

    def setup(self, stage=None):
        dataset = ClinicalDataset(self.csv_path)
        total_size = len(dataset)
        val_size = int(total_size * self.val_split)
        test_size = int(total_size * self.test_split)
        train_size = total_size - val_size - test_size

        self.train_data, self.val_data, self.test_data = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        base_dataset = dataset

        train_labels = [base_dataset[i][1].item() for i in self.train_data.indices]
        val_labels = [base_dataset[i][1].item() for i in self.val_data.indices]
        test_labels = [base_dataset[i][1].item() for i in self.test_data.indices]

        print("\nFinal split sizes:")
        print(f" Train: {len(self.train_data)} | Val: {len(self.val_data)} | Test: {len(self.test_data)}")
        print(f" Train labels: {Counter(train_labels)}")
        print(f" Val labels: {Counter(val_labels)}")
        print(f" Test labels: {Counter(test_labels)}")

    def train_dataloader(self):
        labels = [self.train_data.dataset[i][1] for i in self.train_data.indices]
        class_counts = Counter(labels)
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}

        sample_weights = [class_weights[label] for label in labels]
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=4
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=4
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=4
        )
    