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
from torch_geometric.nn import GAE,VGAE




if __name__ == "__main__":
    # Free cuda memory and clear unused variables
    
    torch.cuda.empty_cache()
    gc.collect()
    # devices = torch.cuda.device_count()
    devices = 1
    # torch.cuda.memory_summary(device=devices, abbreviated=False)

    # Load Dataset
    # data_dir = "/home/ubuntu/date/hdd4/shadow_model_ckpt/mnist/models0"
    data_dir = "/home/jianghaoyu/cifar10_outliers"
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

    