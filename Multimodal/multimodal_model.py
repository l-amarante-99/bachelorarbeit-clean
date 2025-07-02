import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from c_model import ClinicalMLP
from img_model import ResNet3D
from torchmetrics import BinaryAccuracy, BinaryF1Score

class LateFusionModel(pl.LightningModule):
    def __init__(self, clinical_ckpt_path, mri_ckpt_path, input_dim, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()

        if clinical_ckpt_path:
            self.clinical_model = ClinicalMLP.load_from_checkpoint(clinical_ckpt_path, input_dim=input_dim)
        else:
            self.clinical_model = ClinicalMLP(input_dim=input_dim)

        if mri_ckpt_path:
            self.mri_model = ResNet3D.load_from_checkpoint(mri_ckpt_path, in_channels=1)
        else:
            self.mri_model = ResNet3D(in_channels=1)

        # if freeze the clinical and MRI models:
        #for p in self.clinical_model.parameters():
            #p.requires_grad = False
        #for p in self.mri_model.parameters():
            #p.requires_grad = False

        self.clinical_model.model[-1] = nn.Identity()  # Remove the final layer
        self.mri_model.model.fc = nn.Identity()  # Remove the final layer

        self.clinical_model = nn.LayerNorm(32)
        self.mri_model = nn.LayerNorm(512)

        self.classifier = nn.Sequential(
            nn.Linear(32 + 512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

        self.criterion = nn.BCEWithLogitsLoss()
        self.train_acc = BinaryAccuracy()
        self.train_f1 = BinaryF1Score()
        self.val_acc = BinaryAccuracy()
        self.val_f1 = BinaryF1Score()
        self.test_acc = BinaryAccuracy()
        self.test_f1 = BinaryF1Score()

    def forward(self, clinical_x, mri_x):
        c_feat = self.clinical_model(clinical_x)
        m_feat = self.mri_model(mri_x)

        c_feat = self.clinical_norm(c_feat)
        m_feat = self.mri_norm(m_feat)

        combined = torch.cat([c_feat, m_feat], dim=1)
        return self.classifier(combined).squeeze()

    def training_step(self, batch, batch_idx):
        (clinical_x, mri_x), y = batch
        logits = self(clinical_x, mri_x)
        loss = self.criterion(logits, y.float())
        preds = torch.sigmoid(logits) > 0.5
        self.train_acc(preds, y)
        self.train_f1(preds, y)
        self.log('train_loss', loss)
        self.log('train_acc', self.train_acc, prog_bar=True)
        self.log('train_f1', self.train_f1, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        (clinical_x, mri_x), y = batch
        logits = self(clinical_x, mri_x)
        loss = self.criterion(logits, y.float())
        preds = torch.sigmoid(logits) > 0.5
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        self.log('val_loss', loss)
        self.log('val_acc', self.val_acc, prog_bar=True)
        self.log('val_f1', self.val_f1, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        (clinical_x, mri_x), y = batch
        logits = self(clinical_x, mri_x)
        loss = self.criterion(logits, y.float())
        preds = torch.sigmoid(logits) > 0.5
        self.test_acc(preds, y)
        self.test_f1(preds, y)
        self.log('test_loss', loss)
        self.log('test_acc', self.test_acc)
        self.log('test_f1', self.test_f1)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
