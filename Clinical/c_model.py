import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics import BinaryAccuracy, BinaryF1Score

class ClinicalMLP(pl.LightningModule):
    def __init__(self, input_dim, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 1)
        )

        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.train_f1 = BinaryF1Score()
        self.val_f1 = BinaryF1Score()
        self.test_acc = BinaryAccuracy()
        self.test_f1 = BinaryF1Score()

    def forward(self, x):
        x = self.model(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        preds = torch.sigmoid(logits) > 0.5
        self.train_acc(preds, y)
        self.train_f1(preds, y)
        self.log("train_loss", loss)
        self.log("train_acc", self.train_acc, prog_bar=True)
        self.log("train_f1", self.train_f1, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        preds = torch.sigmoid(logits) > 0.5
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc, prog_bar=True)
        self.log("val_f1", self.val_f1, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze(-1)
        preds = (torch.sigmoid(logits) > 0.5).long()
        loss = F.binary_cross_entropy_with_logits(logits, y.float())

        self.test_acc(preds, y)
        self.test_f1(preds, y)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc, prog_bar=True)
        self.log("test_f1", self.test_f1, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=1e-5
        )