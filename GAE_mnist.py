import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
import gc
import numpy as np
import pytorch_lightning as L
import torch
from pytorch_lightning import callbacks
from torch_geometric.data.lightning import LightningDataset
# from torch_geometric.data import lightning
from torch_geometric.nn.models import GIN
from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score
from models import ModelDataset_cifar
from models.gin_1 import GINModule
from torch_geometric.loader import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GAE

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        # in_channels 是特征数量, out_channels * 2 是因为我们有两个GCNConv, 最后我们得到embedding大小的向量
        # cached 因为我们只有一张图
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # 缓存仅用于转导学习
        # self.conv2 = GCNConv(4 * out_channels,2* out_channels, cached=True) # cached only for transductive learning
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        # x = self.conv2(x, edge_index).relu()
        return self.conv2(x, edge_index)



if __name__ == "__main__":
    # Free cuda memory and clear unused variables
    
    torch.cuda.empty_cache()
    gc.collect()
    # devices = torch.cuda.device_count()
    devices = 1
    # torch.cuda.memory_summary(device=devices, abbreviated=False)

    # Load Dataset
    # data_dir = "/home/ubuntu/date/hdd4/shadow_model_ckpt/mnist/models0"
    data_dir = "/home/jianghaoyu/cifar10_benign"
    test_dir = "/home/jianghaoyu/cifar10_outliers"
    dataset = ModelDataset_cifar(data_dir=data_dir)
    test_dataset = ModelDataset_cifar(data_dir=test_dir)
    # print(dataset.num_features)
    dataset = dataset.shuffle()
    test_dataset= test_dataset.shuffle()
    test_dataset = test_dataset
    # test_dataset = dataset[1*len(dataset) // 10:2 * len(dataset) // 10]
    # val_dataset = dataset[1*len(dataset) // 10:2 * len(dataset) // 10]
    train_dataset = dataset
    # train_dataset = dataset[2 * len(dataset) // 10:3 * len(dataset) // 10]
    print("train set: ", len(train_dataset))
    # print("valid set: ", len(val_dataset))
    print("test set: ", len(test_dataset))
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # parameters
    out_channels = 64
    num_features = 262656
    epochs = 100
    
    # model
    model = GAE(GCNEncoder(num_features, out_channels))
    print(model)

    # move to GPU (if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    def train(x,train_pos_edge_index):
        model.train()
        # for data in train_loader:  # Iterate in batches over the training dataset.
        # data = train_dataset[0]
        # data = train_test_split_edges(data)
        # data = data.to(device)
        # out = model.encode(data.x, data.train_pos_edge_index)  # Perform a single forward pass.
        # loss = model.recon_loss(out, data.edge_index)  # Compute the loss.
        # loss.backward()  # Derive gradients.
        # optimizer.step()  # Update parameters based on gradients.
        # optimizer.zero_grad()  # Clear gradients.

        optimizer.zero_grad()
        # model.encode 调用了我们传入的编码器
        z = model.encode(x, train_pos_edge_index)
        # recon_loss 为重构损失
        loss = model.recon_loss(z, train_pos_edge_index)
        #if args.variational:
        #   loss = loss + (1 / data.num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()
        return float(loss)


    def test(x,train_pos_edge_index,pos_edge_index, neg_edge_index):
        model.eval()
        with torch.no_grad():
            z = model.encode(x, train_pos_edge_index)
        # 使用正边和负边来测试模型的准确率
        return model.test(z, pos_edge_index, neg_edge_index)

    for i in range(1):
        data = train_test_split_edges(train_dataset[i])
        x = data.x.to(device)
        train_pos_edge_index = data.train_pos_edge_index.to(device)
        for epoch in range(1, epochs + 1):
            loss = train(x,train_pos_edge_index)
            print(loss)
            # auc 指的是ROC曲线下的面积, ap 指的是平均准确度
            auc, ap = test(x,train_pos_edge_index,data.test_pos_edge_index, data.test_neg_edge_index)
            print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))

    torch.save(model.state_dict(), "GAEmodel_cifar.pt")
    # model_new = GAE(GCNEncoder(num_features, out_channels))
    # #通过 load_state_dict 函数加载参数，torch.load() 函数中重要的一步是反序列化。
    # model_new.load_state_dict(torch.load("GAEmodel.pt"))
    
    