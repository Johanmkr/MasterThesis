import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
import os
import sys

# Add the parent directory of the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
sys.path.append(parent_dir)


# Local imports

import config as cfg
from src.data.custom_data_module import CustomDataModule
from src.data.transforms import Normalise
from src.models.MOUSE import MOUSE

# from src.utils import yamlconfig

# cfg = yamlconfig.YamlConfig("config.yaml")


torch.set_float32_matmul_precision("medium")

if __name__ == "__main__":
    # Create data module
    dm = CustomDataModule(
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        stride=cfg.STRIDE,
        redshifts=cfg.REDSHIFTS,
        seeds=cfg.SEEDS,
        axes=cfg.AXES,
        transform=Normalise(),
        additional_info=False,
        train_test_val_split=cfg.TRAIN_TEST_VAL_SPLIT,
        split_seed=cfg.SPLIT_SEED,
    )
    dm.prepare_data()
    dm.setup(stage="fit")

    # Create model
    model = MOUSE(
        input_channels=dm.dataset.stride,
        ngrid=dm.dataset.ngrid,
        learning_rate=cfg.LEARNING_RATE,
    )

    # Create logger
    logger = TensorBoardLogger(
        save_dir=cfg.LOG_DIR,
        name=cfg.MODEL_NAME,
    )

    # Create profiler
    profiler = PyTorchProfiler(
        output_filename=os.path.join(cfg.LOG_DIR, cfg.MODEL_NAME, "profiler.txt"),
        profile_memory=True,
        group_by_input_shapes=True,
    )

    # Create trainer
    trainer = pl.Trainer(
        accelerator=cfg.ACCELERATOR,
        min_epochs=1,
        max_epochs=cfg.NUM_EPOCHS,
        logger=logger,
        profiler=profiler,
        precision=cfg.PRECISION,
        # limit_train_batches=0.1,
        # limit_val_batches=0.1,
        # limit_test_batches=0.1,
        # limit_predict_batches=0.1,
        # auto_lr_find=True,
    )

    # Train the model
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)
