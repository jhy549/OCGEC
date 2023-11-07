import os

import pytorch_lightning as L
import torch
import torch.nn as nn
from configs import BASE_DIR
from configs.args import init_args, add_yaml_to_args, preprocess_args
from torch.utils.data import random_split
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from .shadow_model import Model9
from utils_1.base import PruningBase
from utils_1 import load_cifar10




class ModelDataset_resnet_v2(Dataset):
    def __init__(self, data_dir: str):
        super().__init__()
        self.data_dir = data_dir
        self.data_path = self.load_data_path()

    def load_data_path(self):
        data_path = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                full_path = os.path.join(root, file)
                # print(full_path)
                data_path.append(full_path)
        return data_path
    
    def load_model_struct_from_path(self, path):
        # get label
        if "benign" in path:
            y = 1
        else:
            y = 0
        label = torch.LongTensor([y])
        args = init_args()
        add_yaml_to_args(args, BASE_DIR/'configs/default.yaml')
        preprocess_args(args)
        # get model
        # print("Preparing load model:", path)
        ordered_dict = torch.load(path)
        # print(type(ordered_dict))
        shadow_model = Model9()
        # from torchvision.models import resnet18
        # shadow_model = resnet18(num_classes=10)
        # print(type(shadow_model))
        shadow_model.load_state_dict(ordered_dict)
        shadow_model = shadow_model.to('cuda:3')

        test_loader, p_test_loader,all,pall = load_cifar10(args)
        P = PruningBase(shadow_model,test_loader, p_test_loader, args,all,pall)
        P.setup()
        edges, features, node_num = P.graph_construction()
        edges= torch.Tensor(edges)
        # print(len(edges[0]))
        return {"edges": edges, "x": features, "y": label}
    
    def transform_model_to_graph_data_by_padding(self, info_dict):
        # TODO
        graph_data = Data(x=info_dict["x"], edge_index=info_dict["edges"],
                          y=info_dict["y"])
        return graph_data

    def get(self, idx: int) -> Data:
        path = self.data_path[idx]
        # print("这是 %s", path)
        # load model 
        struct_info_dict = self.load_model_struct_from_path(path)
        # transform to graph data
        # TODO try other transform method
        graph_data = self.transform_model_to_graph_data_by_padding(struct_info_dict)
        return graph_data
    
    def len(self):
        return len(self.data_path)

   