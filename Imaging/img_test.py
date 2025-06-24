import pytorch_lightning as pl
from img_model import ResNet3D
from img_data import MRIDataModule
import wandb   
from pytorch_lightning.loggers import WandbLogger

checkpoint_path = "PATH-on-Charité-HPC"
meta_csv = "PATH-on-Charité-HPC"
data_dir = "PATH-on-Charité-HPC"
cache_path = "PATH-on-Charité-HPC"

wandb_logger = WandbLogger(
    project="test-binary-classification",
    name="resnet3d-eval-3",
)

model = ResNet3D.load_from_checkpoint(
    checkpoint_path,
    in_channels=1,
    num_classes=2,
    learning_rate=1e-4,
)

dm = MRIDataModule(
    meta_csv=meta_csv,
    data_dir=data_dir,
    cache_dir=cache_path,
    batch_size=2,
    num_workers=4,
)

trainer = pl.Trainer(
    logger=wandb_logger,
    accelerator="auto",
    devices=1,
)

trainer = pl.Trainer(
    logger=wandb_logger,
    accelerator="auto",
    devices=1,
)
trainer.test(model, datamodule=dm)