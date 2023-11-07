import argparse
import sys
from pathlib import Path

import pytorch_lightning as L
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from utils.dataset import CleanDatasetWrapper


def test_ckpt_model_asr():
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = Path(__file__).parent.parent / "data"
    if not DATA_DIR.exists():
        Path.mkdir(DATA_DIR)

    clean_train_set = CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    clean_train_set_wrapper = CleanDatasetWrapper(dataset=clean_train_set)

    clean_train_loader = DataLoader(
        dataset=clean_train_set_wrapper, batch_size=128, shuffle=True
    )

    clean_test_set = CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    clean_test_set_wrapper = CleanDatasetWrapper(dataset=clean_test_set)
    clean_test_loader = DataLoader(dataset=clean_test_set_wrapper, batch_size=128)

    from torchvision.models import mobilenet_v2

    from models.wrapper import ImageModelWrapper

    orig_model = mobilenet_v2(num_classes=10)

    lightning_model = ImageModelWrapper.load_from_checkpoint(
        "path_to" / "models" / "last.ckpt", model=orig_model
    )
    LOG_DIR = BASE_DIR / "logs"
    if not LOG_DIR.exists():
        Path.mkdir(LOG_DIR)

    trainer = L.Trainer(
        devices=1,
        max_epochs=3,
        default_root_dir=LOG_DIR / "ckpt",
        log_every_n_steps=1,
        # accumulate_grad_batches=4,
        # fast_dev_run=True,
        # limit_train_batches=0.01,
        # limit_test_batches=0.01,
    )
    trainer.fit(lightning_model, train_dataloaders=clean_train_loader)
    trainer.test("best", dataloaders=clean_test_loader)
