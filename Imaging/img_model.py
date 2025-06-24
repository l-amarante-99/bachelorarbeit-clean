import torch
import troch.nn.functional as F
import pytorch_lightning as pl
from monai.networks.nets import resnet
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score

class ResNet3D(pl.LightningModule):
    def __init__(self, in_channels=1, learning_rate=1e-4):
        super()__init__()
        self.save_hyperparameters()

        self.model = resnet.resnet18(
            spatial_dims=3,
            in_channels=in_channels,
            num_classes=2,
        )

        self.val_acc = BinaryAccuracy()
        self.val_f1 = BinaryF1Score()
        self.test_acc = BinaryAccuracy()
        self.test_f1 = BinaryF1Score()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, label_smoothing=0.1)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, on_step=False, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = logits.argmax(dim=1)
        loss = F.cross_entropy(logits, y)

        self.val_acc.update(preds, y)
        self.val_f1.update(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        f1 = self.val_f1.compute()
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        self.val_acc.reset()
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = logits.argmax(dim=1)
        loss = F.cross_entropy(logits, y)

        self.test_acc.update(preds, y)
        self.test_f1.update(preds, y)

        self.log("test_loss", loss, prog_bar=True)
        return loss
    
    def on_test_epoch_end(self):
        acc = self.test_acc.compute()
        f1 = self.test_f1.compute()
        self.log("test_acc", acc, prog_bar=True)
        self.log("test_f1", f1, prog_bar=True)
        self.test_acc.reset()
        self.test_f1.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
        )