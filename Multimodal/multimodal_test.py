import pytorch_lightning as pl
from multimodal_model import LateFusionModel
from multimodal_data import MultimodalDataModule
import wandb
from pytorch_lightning.loggers import WandbLogger

checkpoint_path = "PATH-on-Charité-HPC"
merged_csv_path = "PATH-on-Charité-HPC"
data_dir = "PATH-on-Charité-HPC"
cache_path = "PATH-on-Charité-HPC"

wandb_logger = WandbLogger(
    project="test-binary-classification",
    name="multimodal-eval-2",
)

model = LateFusionModel.load_from_checkpoint(
    checkpoint_path,
    clinical_input_dim=27,
    learning_rate=1e-4,
)

dm = MultimodalDataModule(
    merged_csv_path=merged_csv_path,
    data_dir=data_dir,
    cache_path=cache_path,
    batch_size=2,
    num_workers=4,
)

trainer = pl.Trainer(
    logger=wandb_logger,
    accelerator="auto",
    devices=1,
)
trainer.test(model, datamodule=dm)