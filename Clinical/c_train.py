import torch
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import random
from c_data import ClinicalDataModule, ClinicalDataset
from c_model import ClinicalMLP

pl.seed_everything(42)

csv_path = "PATH-on-Charité-HPC"
input_dim = ClinicalDataset(csv_path).X.shape[1]
num_classes = 2

dm = ClinicalDataModule(csv_path, batch_size=64)

model = ClinicalMLP(input_dim=input_dim, lr=1e-4)

wandb_logger = WandbLogger(
    project="clinical-adni-binary-classification",
    name="mlp-lr-1e-4-batch-64-epochs-150-deeper-wrs-csv-5",
    log_model=True,
)

checkpoint_cb = ModelCheckpoint(
    monitor="val_f1",
    mode="max",
    save_top_k=1,
    filename="best-clinical-mlp"
)

trainer = pl.Trainer(
    logger=wandb_logger,
    max_epochs=150,
    accelerator="auto",
    devices=1,
    loggers=wandb_logger,
    callbacks=[checkpoint_cb],
    log_every_n_steps=1,
)
trainer.fit(model, dm)