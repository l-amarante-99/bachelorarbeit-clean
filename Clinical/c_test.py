import torch
import pytorch_lightning as pl
import numpy as np
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from c_data import ClinicalDataModule, ClinicalDataset
from c_model import ClinicalMLP

pl.seed_everything(42)

csv_path = "PATH-on-Charité-HPC"
dm = ClinicalDataModule(csv_path, batch_size=64)
dm.setup("test")

input_dim = ClinicalDataset(csv_path).X.shape[1]

checkpoint_path = "PATH-on-Charité-HPC"
model = ClinicalMLP.load_from_checkpoint(
    checkpoint_path,
    input_dim=input_dim
)

wandb_logger = WandbLogger(
    project="test-binary-classification",
    name="mlp-binary-test-5",
    log_model=False
)

trainer = pl.Trainer(
    logger=wandb_logger,
    accelerator="auto",
    devices=1,
    log_every_n_steps=10,
)
trainer.test(model, datamodule=dm)