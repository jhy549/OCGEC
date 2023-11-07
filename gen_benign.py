import os
import argparse

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
import pytorch_lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from models import BaseModel, Features, Classifier

import settings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate benign models")
    parser.add_argument("--dataset", default="mnist", type=str, required=True, \
        choices=["mnist", "cifar10"])
    parser.add_argument("--max_epochs", default=100, type=int)
    args = parser.parse_args()

    # load data
    if args.dataset == "mnist":
        dataset = MNIST(root=settings.DATASET_ROOT, train=True, \
                                download=True, transform=transforms.ToTensor())
    elif args.dataset == "cifar10":
        dataset = CIFAR10(root=settings.DATASET_ROOT, download=True, transform=transforms.ToTensor())

    # use 20% of training data for validation
    train_set_size = int(len(dataset) * 0.8)
    valid_set_size = len(dataset) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = data.random_split(dataset, [train_set_size, valid_set_size], generator=seed)
    test_set = MNIST(root=settings.DATASET_ROOT, train=False, \
                                download=True, transform=transforms.ToTensor())

    train_loader = DataLoader(train_set)
    valid_loader = DataLoader(valid_set)
    test_loader  = DataLoader(test_set)

    # train model
    model = BaseModel(features=Features(), classifier=Classifier(num_class=10))
    model_save_root_dir = os.path.join(settings.DATASET_ROOT, 
                                       settings.DATASET_DIR[args.dataset], 
                                       "Benign")
    trainer = L.Trainer(default_root_dir=model_save_root_dir, max_epochs=args.max_epochs, \
                         enable_checkpointing=True, # profiler="simple",
                        #  fast_dev_run=True, num_sanity_val_steps=2,
                         limit_train_batches=0.01, limit_val_batches=0.01, limit_test_batches=0.1,
                         callbacks=[EarlyStopping(monitor="val_loss", mode="min")])
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    trainer.test(model, dataloaders=test_loader)