
import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GIN, MLP, global_add_pool,GraphSAGE,ASAPooling,global_max_pool,global_mean_pool
from torchmetrics import AUROC, Accuracy
from torchmetrics.classification import BinaryAccuracy
import numpy as np



class GINModule(L.LightningModule):
    def __init__(self, in_channels: int, out_channels: int,
                 hidden_channels: int = 64, num_layers: int = 3,
                 dropout: float = 0.3, lr: float = 0.001):
        super().__init__()
        self.lr = lr
        self.gnn = GIN(in_channels, hidden_channels, num_layers,
                       dropout=dropout, jk='cat')

        # self.classifier = MLP([hidden_channels, hidden_channels, out_channels],
        #                       norm="batch_norm", dropout=dropout)
        # self.classifier = MetaClassifierOC(4)

        self.num_classes = out_channels
        # self.num_classes = 1
        # self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        # self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        # self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()
        self.val_auroc = AUROC(task="multiclass",num_classes=self.num_classes)
        # self.test_auroc = AUROC(task="multiclass",num_classes=self.num_classes)
        self.test_auroc = AUROC(task="binary")
        self.nu = 0.1
        self.R = torch.tensor(0.0, device=self.device)  # radius R initialized with 0 by default.
        self.c = torch.tensor(0.0, device=self.device) 

        self.gpu = False
        if self.gpu:
            self.cuda(device=2)


    def forward(self, x, edge_index, batch):
        x = self.gnn(x, edge_index)
        x = global_add_pool(x, batch)
        # print(x)
        dist = torch.sum((x - self.c) ** 2, dim=1)
        scores = dist - self.R ** 2
        print(scores)
        print(scores.size())
        # x = self.classifier.forward(x)
        # print("这是池化x",x.shape)
        return scores

    def training_step(self, data, batch_idx):

        y_hat = self(data.x, data.edge_index, data.batch)
        print("这是y_hat",y_hat)
        loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(y_hat), y_hat))
        
        # loss = F.cross_entropy(y_hat, data.y)
        # y_hat.view(4)
        # y_hat = torch.unsqueeze(y_hat, dim=0)
        # y_hat = y_hat.expand(4)
        # print(y_hat)
        # print("这是datay",data.y)
        # data.y = torch.squeeze(data.y)
        # print(data.y[0].unsqueeze(dim=-1))

        # self.preds.append(y_hat.item())
        # self.labs.append(data.y[0].unsqueeze(dim=-1).item())
        # self.r = np.percentile(self.preds, 100*self.v)
        # self.train_acc(y_hat, data.y)
        self.train_acc(y_hat.softmax(dim=-1), data.y)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=y_hat.size(0))
        print("这是loss",loss)
        return loss

    def validation_step(self, data, batch_idx):
        y_hat = self(data.x, data.edge_index, data.batch)
        # print("这是y_hat",y_hat)
        # y_hat = torch.unsqueeze(y_hat, dim=0)
        # print("vali_y:",y_hat)
        # # y_hat = y_hat.expand(4)
        # print("y的size",y_hat.size())
        # print("vali_y",data.y.size())
        self.val_acc(y_hat.softmax(dim=-1), data.y)
        # self.val_auroc(y_hat, data.y[0].unsqueeze(dim=-1))
        # self.val_auroc(y_hat.softmax(dim=-1), data.y[0].unsqueeze(dim=-1))
        self.log('val_acc', self.val_acc, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=y_hat.size(0))
        # self.log('val_auroc', self.val_auroc, prog_bar=True, on_step=False,
        #          on_epoch=True, batch_size=y_hat.size(0))

    def test_step(self, data, batch_idx):
        y_hat = self(data.x, data.edge_index, data.batch)
        # y_hat = torch.unsqueeze(y_hat, dim=0)
        # print("test_y:",data.y)
        # print(y_hat.softmax(dim=-1).size())
        # print(data.y[0].unsqueeze(dim=-1).size())
        self.test_acc(y_hat.softmax(dim=-1), data.y)
        self.test_auroc(y_hat.softmax(dim=-1), data.y)
        self.log('test_acc', self.test_acc, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=y_hat.size(0))
        self.log('test_auroc', self.test_auroc, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=y_hat.size(0))
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def roc(self):
        # preds = np.array(self.preds)
        # labs = np.array(self.labs)
        # self.preds = self.preds.cpu()
        # self.labss = self.labs.cpu()
        # print(type(preds))

        return self.preds,self.labs