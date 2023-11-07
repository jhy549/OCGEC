import os

import pytorch_lightning as L
import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from .shadow_model import Model0


class ModelDataset(Dataset):
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

        # get model
        # print("Preparing load model:", path)
        ordered_dict = torch.load(path)
        # print(type(ordered_dict))
        shadow_model = Model0()
        # print(type(shadow_model))
        shadow_model.load_state_dict(ordered_dict)
        # print(shadow_model)

        # get model struct info
        with torch.no_grad():
            # nodes_feat 512 * 513
            nodes_feat = []
            cnt = 0
            # get conv1 nodes
            conv1 = {}
            for weight in shadow_model.conv1.weight:  
                # print(weight.shape)  
                pad = nn.ZeroPad2d(padding=(254, 254, 253, 254))
                feat = pad(weight[0])
                # print(feat.shape)
                conv1[cnt] = feat
                nodes_feat.append(torch.reshape(feat, (1, 262656))[0])
                cnt += 1
            # print(conv1[0].shape)
            # get conv2 nodes
            conv2 = {}
            for weight in shadow_model.conv2.weight:
                pad = nn.ZeroPad2d(padding=(254, 254, 253, 254))
                feat = pad(weight[0])
                conv2[cnt] = feat
                nodes_feat.append(torch.reshape(feat, (1, 262656))[0])
                cnt += 1
            # print(conv2[16].shape)
            # get conv1 -> conv2 edges
            conv1_2 = []
            for src in conv1.keys():
                for dst in conv2.keys():
                    conv1_2.append([src, dst])

            # get fc node
            fc_index = cnt
            cnt += 1
            fc_node = torch.concat([shadow_model.fc.weight, shadow_model.fc.bias.reshape(512, 1)], 1)
            # print(fc_node.shape)
            nodes_feat.append(torch.reshape(fc_node, (1, 262656))[0])
            # print(fc_node.shape)

            # get conv2 -> fc edges
            conv2_fc = []
            for src in conv2.keys():
                conv2_fc.append([src, fc_index])

            # get output node
            out_index = cnt
            cnt += 1
            out = torch.concat([shadow_model.output.weight, shadow_model.output.bias.reshape(10, 1)], 1)
            pad = nn.ZeroPad2d(padding=(0, 0, 251, 251))
            out_node = pad(out)
            nodes_feat.append(torch.reshape(out_node, (1, 262656))[0])

            # print(out_node.shape)

            # get fc -> output edge
            fc_out_edge = [[fc_index, out_index]]

            # get all nodes
            nodes_feat = torch.stack(nodes_feat)
            # print(nodes_feat.shape)
            # get all edges
            all_edges = torch.tensor(conv1_2 + conv2_fc + fc_out_edge).t()
            # print(all_edges)
            # g = Graph(edge_index=all_edges, x=nodes_feat, y=label)

        return {"edges": all_edges, "x": nodes_feat, "y": label}
    
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

   