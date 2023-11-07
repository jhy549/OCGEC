import gc
import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
from barbar import Bar
from pytorch_lightning import callbacks
from torch_geometric.data.lightning import LightningDataset
# from torch_geometric.data import lightning
from torch_geometric.nn.models import GIN,MLP
from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score
from models import ModelDataset_audio
from models.gin_1 import GINModule
from torch_geometric.loader import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_add_pool

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')



def eval(net, c, dataloader, device):
    """Testing the Deep SVDD model"""

    scores = []
    labels = []
    net.eval()
    print('Testing...')
    with torch.no_grad():
        for data in dataloader:
            # x = data.x.reshape(-1).to(device)
            data = data.to(device)
            # x = x.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            z = net(data.x, data.edge_index, data.batch)
            score = torch.sum((z - c) ** 2, dim=-1)
            # print("这是score",score)
            # print("这是y",data.y)
            # print(score.shape)
            scores.append(score.unsqueeze(0).detach().cpu())
            labels.append(data.y.unsqueeze(0).cpu())
    labels, scores = torch.cat(labels).numpy(), torch.cat(scores).numpy()
    print('ROC AUC score: {:.2f}'.format(roc_auc_score(labels, scores)*100))
    return labels, scores


# class network(nn.Module):
#     def __init__(self, z_dim=1035):
#         super(network, self).__init__()
#         self.pool = nn.MaxPool2d(2, 2)

#         self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
#         self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
#         self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
#         self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
#         self.fc1 = nn.Linear(4 * 7 * 7, z_dim, bias=False)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.pool(F.leaky_relu(self.bn1(x)))
#         x = self.conv2(x)
#         x = self.pool(F.leaky_relu(self.bn2(x)))
#         x = x.view(x.size(0), -1)
#         return self.fc1(x)

# class network(nn.Module):
#     def __init__(self, z_dim=1035):
#         super(network, self).__init__()

#         self.fc1 = nn.Linear(z_dim, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 32)

#     def forward(self, x):
#         x = F.leaky_relu(self.fc1(x))
#         x = F.leaky_relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

class network(L.LightningModule):
    def __init__(self, in_channels: int, out_channels: int,
                 hidden_channels: int = 64, num_layers: int = 3,
                 dropout: float = 0.5, lr: float = 0.01):
        super().__init__()
        self.lr = lr
        self.cuda()
        self.gnn = GIN(in_channels, hidden_channels, num_layers,
                       dropout=dropout, jk='cat')

        self.classifier = MLP([hidden_channels, hidden_channels, out_channels],
                              norm="batch_norm", dropout=dropout)
        # self.classifier = MetaClassifierOC(4)

        self.num_classes = out_channels
        # self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        # self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        # self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        # self.val_auroc = AUROC(task="multiclass",num_classes=self.num_classes)
        # self.test_auroc = AUROC(task="multiclass",num_classes=self.num_classes)
    def forward(self, x, edge_index, batch):
        x = self.gnn(x, edge_index)
        x = global_add_pool(x, batch)
        # print("这是池化x",x.shape)
        # x = self.classifier.forward(x)
        # print("这是池化x",x.shape)
        return x
    


class TrainerDeepSVDD:
    def __init__(self ,device,epoch):
        self.epoch = epoch
        # self.train_loader, self.test_loader = data
        self.device = device

    def set_c(self,dataloader, eps=0.1):
        """Initializing the center for the hypersphere"""
        # z_ = []
        # with torch.no_grad():
        #     for data in dataloader:
        #         # print(data)
        #         x = data.x.reshape(-1).to(self.device)

        #         z_.append(x.detach())
        # z_ = torch.cat(z_)
        # c = torch.mean(z_, dim=0)
        # c[(abs(c) < eps) & (c < 0)] = -eps
        # c[(abs(c) < eps) & (c > 0)] = eps
        c = eps
        return c
    
    def train(self,dataloader):
        net = network(in_channels=102, hidden_channels=64, out_channels=2).to(self.device)
        c = self.set_c(dataset)   
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001,
                               weight_decay=0.5e-6)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                    milestones=[50], gamma=0.1)

        net.train()
        for epoch in range(self.epoch):
            total_loss = 0
            for data in dataloader:
                # x = data.x.reshape(-1).to(self.device)
                data = data.to(device)
                # x = x.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                optimizer.zero_grad()
                z = net(data.x, data.edge_index, data.batch)
                loss = torch.mean(torch.sum((z - c) ** 2, dim=-1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            scheduler.step()
            print('Training Deep SVDD... Epoch: {}, Loss: {:.3f}'.format(
                   epoch, total_loss/len(dataloader)))
        self.net = net
        self.c = c










if __name__ == "__main__":
    # Free cuda memory and clear unused variables
    
    torch.cuda.empty_cache()
    gc.collect()
    # devices = torch.cuda.device_count()
    # torch.cuda.memory_summary(device=devices, abbreviated=False)

    # Load Dataset
    # data_dir = "/home/ubuntu/date/hdd4/shadow_model_ckpt/mnist/models0"
    data_dir = "/home/jianghaoyu/audio_benign"
    test_dir = "/home/jianghaoyu/audio_outliers"
    dataset = ModelDataset_audio(data_dir=data_dir)
    test_dataset = ModelDataset_audio(data_dir=test_dir)
    # print(dataset.num_features)
    dataset = dataset.shuffle()
    test_dataset= test_dataset.shuffle()
    test_dataset = test_dataset
    # test_dataset = dataset[1*len(dataset) // 10:2 * len(dataset) // 10]
    # val_dataset = dataset[1*len(dataset) // 10:2 * len(dataset) // 10]
    train_dataset = dataset[2 * len(dataset) // 10:3 * len(dataset) // 10]
    print("train set: ", len(train_dataset))
    # print("valid set: ", len(val_dataset))
    print("test set: ", len(test_dataset))
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    deep_SVDD = TrainerDeepSVDD(device,5)
    deep_SVDD.train(train_loader)
    labels, scores = eval(deep_SVDD.net, deep_SVDD.c, test_loader, device)
    # labels, scores = eval(deep_SVDD.net, deep_SVDD.c, train_loader, device)
