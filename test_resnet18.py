import gc
import numpy as np
import pytorch_lightning as L
import torch
from pytorch_lightning import callbacks
from torch_geometric.data.lightning import LightningDataset
# from torch_geometric.data import lightning
from torch_geometric.nn.models import GIN
from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score
from models import ModelDataset_resnet
from models import ModelDataset
from models.gin_3 import GINModule

if __name__ == "__main__":
    # Free cuda memory and clear unused variables
    
    torch.cuda.empty_cache()
    gc.collect()
    # devices = torch.cuda.device_count()
    devices = 1
    # torch.cuda.memory_summary(device=devices, abbreviated=False)

    # Load Dataset
    # data_dir = "/home/ubuntu/date/hdd4/shadow_model_ckpt/mnist/models0"
    data_dir = "/home/jianghaoyu/Meta-Nerual-Trojan-Detection/resnet/cifar10/1/"
    # data_dir = "/home/jianghaoyu/jhy_models"
    # test_dir = "/home/jianghaoyu/audio_outliers"
    dataset = ModelDataset_resnet(data_dir=data_dir)
    # test_dataset = ModelDataset(data_dir=test_dir)
    # print(dataset.num_features)
    # dataset = dataset.shuffle()
    # test_dataset= test_dataset.shuffle()
    # test_dataset = test_dataset
    # test_dataset = dataset[1*len(dataset) // 10:2 * len(dataset) // 10]
    # val_dataset = dataset[1*len(dataset) // 10:2 * len(dataset) // 10]
    # train_dataset = dataset[2 * len(dataset) // 10:6 * len(dataset) // 10]
    train_dataset= dataset
    # print(torch.eq(train_dataset[0].edge_index,train_dataset[2].edge_index))
    # print(train_dataset[0].edge_index)
    print(train_dataset[0])
    # print("train set: ", len(train_dataset))
    # print("valid set: ", len(val_dataset))
    # print("test set: ", len(test_dataset))