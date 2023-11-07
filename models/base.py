from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import pytorch_lightning as L
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.datasets import MNIST
import torchmetrics

from attack import InputModifyAttack

class Features(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
    
    def forward(self, x):
        return self.features(x)


class Classifier(nn.Module):
    
    def __init__(self, num_class) -> None:
        super().__init__()
        self.num_class = num_class
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 512),
            nn.Linear(512, self.num_class)
        )
    
    def forward(self, x):
        return self.classifier(x)


class BaseModel(L.LightningModule):
    
    def __init__(self, features, classifier) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.features = features
        self.classifier = classifier
        self.accuracy = torchmetrics.Accuracy(task="multiclass", 
                                              num_classes=classifier.num_class)

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> Optional[int]:
        return super().on_train_batch_start(batch, batch_idx)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop
        x, y = batch
        feats = self.features(x)
        logits = self.classifier(feats)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)
        self.accuracy(logits, y)
        self.log("train_acc_step", self.accuracy)
        return loss

    def training_epoch_end(self, outputs) -> None:
        self.log("train_acc_epoch", self.accuracy)
        return super().training_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        # test loop
        x, y = batch
        feats = self.features(x)
        logits = self.classifier(feats)
        loss = F.cross_entropy(logits, y)
        self.log("test_loss", loss)
        self.accuracy(logits, y)
        self.log("test_acc", self.accuracy)

    def validation_step(self, batch, batch_idx):
        # val loop
        x, y = batch
        feats = self.features(x)
        logits = self.classifier(feats)
        loss = F.cross_entropy(logits, y)
        self.log("val_loss", loss)

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

class MaliciousDataLoader(DataLoader):

    def set_attack(self, attack):
        self.attack = attack
    
    # append attack on the returned data
    def __next__(self) -> Any:
        try:
            return self.attack(super().__next__())
        except:
            raise ValueError("attack method not set before.")
        
# class CustomedMNIST(MNIST):
    
#     def __init__(self, root: str, save_folder: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False) -> None:
#         self.save_folder = save_folder
#         super().__init__(root, train, transform, target_transform, download)
    
#     @property
#     def raw_folder(self) -> str:
#         return self.save_folder