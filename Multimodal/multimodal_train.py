import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from multimodal_data import MultimodalDataModule
from multimodal_model import LateFusionModel
import os

merged_csv_path = "PATH-on-Charité-HPC"
data_dir = "PATH-on-Charité-HPC"
cache_path = "PATH-on-Charité-HPC"

dm = MultimodalDataModule(
    merged_csv_path=merged_csv_path,
    data_dir=data_dir,
    cache_path=cache_path,
    batch_size=8,
    num_workers=4,
)

dm.setup()

wandb_logger = WandbLogger(
    project="multimodal-adni-binary-classification",
    name="latefusion-batch=8-epochs=100-lr=1e-4-layernorm",
    log_model=True,
)

model = LateFusionModel(
    clinical_ckpt_path=None,
    mri_ckpt_path=None,
    input_dim=len(dm.clinical_features),
    learning_rate=1e-4,
)

checkpoint_cb = ModelCheckpoint(
    monitor="val_f1",
    mode="max",
    save_top_k=1,
    verbose=True,
    filename="latefusion-best"
)

trainer = pl.Trainer(
    logger=wandb_logger,
    max_epochs=100,
    accelerator="auto",
    callbacks=[checkpoint_cb]
    log_every_n_steps=5
)
trainer.fit(model, datamodule=dm)


