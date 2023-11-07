
import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GIN, MLP, global_add_pool
from torchmetrics import AUROC, Accuracy
import numpy as np

class GINModule(L.LightningModule):
    def __init__(self, in_channels: int, out_channels: int,
                 hidden_channels: int = 64, num_layers: int = 3,
                 dropout: float = 0.0, lr: float = 0.001):
        super().__init__()
        self.lr = lr
        self.gnn = GIN(in_channels, hidden_channels, num_layers,
                       dropout=dropout, jk='cat')

        self.classifier = MLP([hidden_channels, 128, out_channels],
                              norm="batch_norm", dropout=dropout)
        # self.classifier = MetaClassifierOC(4)

        self.num_classes = out_channels
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_auroc = AUROC(task="multiclass",num_classes=self.num_classes)
        self.test_auroc = AUROC(task="multiclass",num_classes=self.num_classes)
    def forward(self, x, edge_index, batch):
        x = self.gnn(x, edge_index)
        x = global_add_pool(x, batch)
        # print("这是池化x",x.shape)
        x = self.classifier(x)
        # print("这是池化x",x.shape)
        return x

    def training_step(self, data, batch_idx):
        y_hat = self(data.x, data.edge_index, data.batch)
        # print("这是y_hat",y_hat.shape)
        loss = F.cross_entropy(y_hat, data.y)
        # loss=MetaClassifierOC.loss(y_hat)
        # print("这是y_hat",y_hat)
        # print("这是y",data.y)
        self.train_acc(y_hat.softmax(dim=-1), data.y)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=y_hat.size(0))
        # print("这是loss",loss)
        return loss

    def validation_step(self, data, batch_idx):
        y_hat = self(data.x, data.edge_index, data.batch)
        # print("这是y_hat",y_hat)
        self.val_acc(y_hat.softmax(dim=-1), data.y)
        self.val_auroc(y_hat.softmax(dim=-1), data.y)
        self.log('val_acc', self.val_acc, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=y_hat.size(0))
        self.log('val_auroc', self.val_auroc, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=y_hat.size(0))

    def test_step(self, data, batch_idx):
        y_hat = self(data.x, data.edge_index, data.batch)
        # print("这是y_hat",y_hat)
        self.test_acc(y_hat.softmax(dim=-1), data.y)
        self.test_auroc(y_hat.softmax(dim=-1), data.y)
        self.log('test_acc', self.test_acc, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=y_hat.size(0))
        self.log('test_auroc', self.test_auroc, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=y_hat.size(0))
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)