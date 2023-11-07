import os

import pytorch_lightning as L
import torch
import torch.nn as nn

from torch.utils.data import random_split
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from .shadow_model import Model8


class ModelDataset_nlp(Dataset):
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
        shadow_model = Model8()
        # print(type(shadow_model))
        shadow_model.load_state_dict(ordered_dict)

        # print(shadow_model)
        # for m in shadow_model.modules():
        #     print(m)
        # for name, param in shadow_model.named_parameters():
        #     print(name,param.shape)
        
        # get model struct info
        with torch.no_grad():
            # nodes_feat 102
            nodes_feat = []
            cnt = 0

            conv1_3 = {}
            for weight in shadow_model.conv1_3.weight:  
                # print(weight.reshape(-1).shape)  
                pad = nn.ZeroPad2d(padding=(300, 300))
                feat = pad(weight.reshape(-1))
                # print(feat.shape)
                conv1_3[cnt] = feat
                nodes_feat.append(torch.reshape(feat, (1, 1500))[0])
                cnt += 1

            conv1_4 = {}
            for weight in shadow_model.conv1_4.weight:  
                # print(weight.reshape(-1).shape)  
                pad = nn.ZeroPad2d(padding=(150, 150))
                feat = pad(weight.reshape(-1))
                # print(feat.shape)
                conv1_4[cnt] = feat
                nodes_feat.append(torch.reshape(feat, (1, 1500))[0])
                cnt += 1
            
            conv1_5 = {}
            for weight in shadow_model.conv1_5.weight:  
                # print(weight.reshape(-1).shape)  
                pad = nn.ZeroPad2d(padding=(0, 0))
                feat = pad(weight.reshape(-1))
                # print(feat.shape)
                conv1_3[cnt] = feat
                nodes_feat.append(torch.reshape(feat, (1, 1500))[0])
                cnt += 1
           
            out_index = cnt
            cnt += 1
            out = torch.concat([shadow_model.output.weight, shadow_model.output.bias.reshape(1,1)], 1)
            # print(out.shape)
            pad = nn.ZeroPad2d(padding=(599, 600))
            out_node = pad(out)
            nodes_feat.append(torch.reshape(out_node, (1, 1500))[0])

            conv3_4 = []
            for src in conv1_3.keys():
                for dst in conv1_4.keys():
                    conv3_4.append([src, dst])
            conv4_5 = []
            for src in conv1_4.keys():
                for dst in conv1_5.keys():
                    conv4_5.append([src, dst])

            conv5_out = []
            for src in conv1_5.keys():
                conv5_out.append([src, out_index])
   
            # ih_hh_l0 = []
            # for src in ih_l0.keys():
            #     for dst in hh_l0.keys():
            #         ih_hh_l0.append([src, dst])

            # hh_ih_l1 = []
            # for src in hh_l0.keys():
            #     for dst in ih_l1.keys():
            #         hh_ih_l1.append([src, dst])

            # ih_hh_l1 = []
            # for src in ih_l1.keys():
            #     for dst in hh_l1.keys():
            #         ih_hh_l1.append([src, dst])
            # hh_l1_att = []
            # for src in hh_l1.keys():
            #     for dst in att.keys():
            #         ih_hh_l1.append([src, dst])

            # att_out = []
            # for src in att.keys():
            #     for dst in out.keys():
            #         att_out.append([src, dst])

            # get all nodes
            nodes_feat = torch.stack(nodes_feat)
            # print(nodes_feat.shape)
            # get all edges
            # all_edges = torch.tensor(ih_hh_l0 + hh_ih_l1 + ih_hh_l1 + hh_l1_att + att_out).t()
            all_edges = torch.tensor(conv3_4+conv4_5+ conv5_out).t()
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

   