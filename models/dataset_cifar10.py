import os
import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from .shadow_model import Model5


class ModelDataset_cifar(Dataset):
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
        shadow_model = Model5()
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
                pad = nn.ZeroPad2d(padding=(0, 0, 0, 0))
                # pad = nn.ZeroPad2d(padding=(255, 255, 254, 255))

                feat = pad(weight[0])
                # print(feat.shape)
                conv1[cnt] = feat
                nodes_feat.append(torch.reshape(feat, (1, 9))[0])
                cnt += 1
            # print(conv1[0].shape)
            # get conv2 nodes
            conv2 = {}
            for weight in shadow_model.conv2.weight:
                # print(weight[0])
                # pad = nn.ZeroPad2d(padding=(255, 255, 254, 255))
                pad = nn.ZeroPad2d(padding=(0, 0, 0, 0))
                feat = pad(weight[0])
                conv2[cnt] = feat
                nodes_feat.append(torch.reshape(feat, (1, 9))[0])
                cnt += 1
            conv3 = {}
            for weight in shadow_model.conv3.weight:
                # pad = nn.ZeroPad2d(padding=(255, 255, 254, 255))
                pad = nn.ZeroPad2d(padding=(0, 0, 0, 0))
                feat = pad(weight[0])
                conv3[cnt] = feat
                nodes_feat.append(torch.reshape(feat, (1, 9))[0])
                cnt += 1
            conv4 = {}
            for weight in shadow_model.conv2.weight:
                # pad = nn.ZeroPad2d(padding=(255, 255, 254, 255))
                pad = nn.ZeroPad2d(padding=(0, 0, 0, 0))
                feat = pad(weight[0])
                conv4[cnt] = feat
                # nodes_feat.append(torch.reshape(feat, (1, 262656))[0])
                nodes_feat.append(torch.reshape(feat, (1, 9))[0])

                cnt += 1
            # print(conv2[16].shape)
            # get conv1 -> conv2 edges
            conv1_2 = []
            for src in conv1.keys():
                for dst in conv2.keys():
                    conv1_2.append([src, dst])
            conv2_3 = []
            for src in conv2.keys():
                for dst in conv3.keys():
                    conv2_3.append([src, dst])
            conv3_4 = []
            for src in conv3.keys():
                for dst in conv4.keys():
                    conv1_2.append([src, dst])
            # get linear node
            linear_index = cnt
            cnt += 1
            linear_node = torch.concat([shadow_model.linear.weight, shadow_model.linear.bias.reshape(256, 1)], 1)

            # 将张量压缩成[3, 3]大小的张量
            compressed_tensor = torch.zeros((3, 3))

            # 计算每个压缩张量元素的值
            for i in range(3):
                for j in range(3):
                    row_start = i * 85
                    row_end = (i + 1) * 85
                    col_start = j * 171
                    col_end = (j + 1) * 171
                    compressed_tensor[i, j] = torch.mean(linear_node[row_start:row_end, col_start:col_end])
            # print(compressed_tensor)
            pad = nn.ZeroPad2d(padding=(0, 0, 0, 0))
            # pad = nn.ZeroPad2d(padding=(0, 0, 128, 128))
            linear_node = pad(compressed_tensor)
            # print(linear_node.shape)
            nodes_feat.append(torch.reshape(linear_node, (1, 9))[0])
            # get fc node
            fc_index = cnt
            cnt += 1
            fc_node = torch.concat([shadow_model.fc.weight, shadow_model.fc.bias.reshape(256, 1)], 1)
            tensor = torch.zeros((3, 3))

            # 计算每个压缩张量元素的值
            for i in range(3):
                for j in range(3):
                    row_start = i * 85
                    row_end = (i + 1) * 85
                    col_start = j * 85
                    col_end = (j + 1) * 85
                    tensor[i, j] = torch.mean(fc_node[row_start:row_end, col_start:col_end])
            
            # print("这是fc",tensor.size())
            # pad = nn.ZeroPad2d(padding=(128, 128, 128, 128))
            pad = nn.ZeroPad2d(padding=(0, 0, 0, 0))
            fc_node = pad(tensor)
            # print(fc_node.shape)
            
            nodes_feat.append(torch.reshape(fc_node, (1, 9))[0])

            
            # get conv4 -> linear edges
            conv4_linear = []
            for src in conv4.keys():
                conv4_linear.append([src, linear_index])
            #get linear ->  fc dges
            linear_fc_edge = [[linear_index,fc_index]]

            # get output node
            out_index = cnt
            cnt += 1
            out = torch.concat([shadow_model.output.weight, shadow_model.output.bias.reshape(10, 1)], 1)
            tensor_1 = torch.zeros((3, 3))

            # 计算每个压缩张量元素的值
            for i in range(3):
                for j in range(3):
                    row_start = i * 3
                    row_end = (i + 1) * 3
                    col_start = j * 85
                    col_end = (j + 1) * 85
                    tensor_1[i, j] = torch.mean(out[row_start:row_end, col_start:col_end])
            # print(tensor_1.size())
            pad = nn.ZeroPad2d(padding=(0, 0, 0, 0))

            # pad = nn.ZeroPad2d(padding=(128, 128, 251, 251))
            out_node = pad(tensor_1)
            # print(out_node.shape)
            nodes_feat.append(torch.reshape(out_node, (1, 9))[0])

            # print(out_node.shape)

            # get fc -> output edge
            fc_out_edge = [[fc_index, out_index]]

            # get all nodes
            nodes_feat = torch.stack(nodes_feat)
            # print(nodes_feat.shape)
            # get all edges
            all_edges = torch.tensor( conv1_2 + conv2_3 + conv3_4 + conv4_linear + linear_fc_edge + fc_out_edge).t()
            # print(all_edges)
            # g = Graph(edge_index=all_edges, x=nodes_feat, y=label)

        return {"edges": all_edges, "x": nodes_feat, "y": label}
    
    def transform_model_to_graph_data_by_padding(self, info_dict):
        # TODO
        # edge_attr = torch.ones((info_dict["edges"].size(0),info_dict["edges"].size(1)))
        graph_data = Data(x=info_dict["x"], edge_index=info_dict["edges"],
                          y=info_dict["y"])
                          #,edge_attr=edge_attr)
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

   