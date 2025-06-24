from pytorch_lightning.loggers import WandbLogger
from img_data import MRIDataModule
from img_model import ResNet3D
import pytorch_lightning as pl
from pytorch_lighting.callbacks import ModelCheckpoint
import torch

torch.set_float32_matmul_precision("medium")

WandbLogger = WandbLogger(
    project="mri-resnet-3D-model",
    name="DA2-lr-1e-3-120-epochs-batchsize-8-dim-128",
)

dm = MRIDataModule(
    meta_csv="PATH-on-Charité-HPC",
    data_dir="PATH-on-Charité-HPC",
    cache_dir="PATH-on-Charité-HPC",
    batch_size=8,
    num_workers=4,
)

model = ResNet3D(in_channels=1)

trainer = pl.Trainer(
    logger=WandbLogger,
    max_epochs=120,
    accelerator="auto",
    devices=1,
    log_every_n_steps=1,
    callbacks=[
        ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="best-resnet18-3d",
        )
    ],
)

trainer.fit(model, datamodule=dm)