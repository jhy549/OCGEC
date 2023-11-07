import os

import pytorch_lightning as L
import torch
import torch.nn as nn

from torch.utils.data import random_split
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from .shadow_model import Model7

def window(ip_tensor,ip_size,out_size):
    compressed_tensor = torch.empty(out_size)
    if ip_size % out_size ==0:
        window_size = ip_size//out_size
        stride = ip_size//out_size
    else:
        window_size = ip_size//out_size + 1
        stride = ip_size//out_size + 1
    for i in range(0,ip_size , stride):
    # 提取滑窗区域
        window = ip_tensor[i:i+window_size]
    
    # 计算滑窗区域的平均值
        average = torch.mean(window)
    
    # 将平均值添加到压缩张量中
        compressed_tensor[i//stride] = average
    return compressed_tensor



class ModelDataset_resnet(Dataset):
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
        shadow_model = Model7()
        # from torchvision.models import resnet18
        # shadow_model = resnet18(num_classes=10)
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

            # for i,weight in enumerate (shadow_model.layer4[0].left[0].weight):
            conv1 = {}
            for weight in shadow_model.conv1[0].weight:
                pad = nn.ZeroPad2d(padding=(101, 0))
                feat = pad(weight.flatten())
                conv1[cnt] = feat
                # print(feat.size())
                nodes_feat.append(torch.reshape(feat, (1, 128))[0])
                cnt += 1
            
            # layer1 = {}
            # # lin = nn.Linear(576,128)   #卷积、池化、还是全连接
            # for weight in shadow_model.layer1[0].left[0].weight:
            #     x = weight.reshape(-1) #64*3*3=576
            #     x = window(x,576,128)
            #     layer1[cnt] = x
            #     nodes_feat.append(torch.reshape(x, (1, 128))[0])
            #     cnt += 1
            # for weight in shadow_model.layer1[0].left[3].weight:
            #     x = weight.reshape(-1) #64*3*3=576
            #     x = window(x,576,128)
            #     layer1[cnt] = x
            #     nodes_feat.append(torch.reshape(x, (1, 128))[0])
            #     cnt += 1
            # for weight in shadow_model.layer1[1].left[0].weight:
            #     x = weight.reshape(-1) #64*3*3=576
            #     x = window(x,576,128)
            #     layer1[cnt] = x
            #     nodes_feat.append(torch.reshape(x, (1, 128))[0])
            #     cnt += 1
            # for weight in shadow_model.layer1[1].left[3].weight:
            #     x = weight.reshape(-1) #64*3*3=576
            #     x = window(x,576,128)
            #     layer1[cnt] = x
            #     nodes_feat.append(torch.reshape(x, (1, 128))[0])
            #     cnt += 1
            #     # print(lin(weight.reshape(64,3*3)).size())
            # layer2 = {}
            # # lin_2 = nn.Linear(128*9,128)
            # for weight in shadow_model.layer2[0].left[0].weight:
            #     x = weight.reshape(-1) #64*3*3=576
            #     x = window(x,576,128)
            #     layer2[cnt] = x
            #     nodes_feat.append(torch.reshape(x, (1, 128))[0])
            #     cnt += 1
            # for weight in shadow_model.layer2[0].left[3].weight:
            #     x = weight.reshape(-1) #1152
            #     x = window(x,1152,128)
            #     layer2[cnt] = x
            #     nodes_feat.append(torch.reshape(x, (1, 128))[0])
            #     cnt += 1
            # for weight in shadow_model.layer2[0].shortcut[0].weight:
            #     x = weight.reshape(-1)
            #     pad = nn.ZeroPad2d(padding=(64, 0))
            #     x = pad(x)
            #     layer2[cnt] = x
            #     nodes_feat.append(torch.reshape(x, (1, 128))[0])
            #     cnt += 1
            # for weight in shadow_model.layer2[1].left[0].weight:
            #     x = weight.reshape(-1) #1152
            #     x = window(x,1152,128)
            #     layer2[cnt] = x
            #     nodes_feat.append(torch.reshape(x, (1, 128))[0])
            #     cnt += 1
            # for weight in shadow_model.layer2[1].left[3].weight:
            #     x = weight.reshape(-1) #1152
            #     x = window(x,1152,128)
            #     layer2[cnt] = x
            #     nodes_feat.append(torch.reshape(x, (1, 128))[0])
            #     cnt += 1

            layer3 = {}
            # lin_3 = nn.Linear(2304,128)
            for weight in shadow_model.layer3[0].left[0].weight:
                x = weight.reshape(-1) #1152
                x = window(x,1152,128)
                layer3[cnt] = x
                nodes_feat.append(torch.reshape(x, (1, 128))[0])
                cnt += 1
            for weight in shadow_model.layer3[0].left[3].weight:
                x = weight.reshape(-1) #2304
                x = window(x,2304,128)
                layer3[cnt] = x
                nodes_feat.append(torch.reshape(x, (1, 128))[0])
                cnt += 1
            for weight in shadow_model.layer3[0].shortcut[0].weight:
                x = weight.reshape(-1) #128
                layer3[cnt] = x
                nodes_feat.append(torch.reshape(x, (1, 128))[0])
                cnt += 1
            for weight in shadow_model.layer3[1].left[0].weight:
                x = weight.reshape(-1) #2304
                x = window(x,2304,128)
                layer3[cnt] = x
                nodes_feat.append(torch.reshape(x, (1, 128))[0])
                cnt += 1
            for weight in shadow_model.layer3[1].left[3].weight:
                x = weight.reshape(-1) #2304
                x = window(x,2304,128)
                layer3[cnt] = x
                nodes_feat.append(torch.reshape(x, (1, 128))[0])
                cnt += 1

            layer4 = {}
            # lin_4 = nn.Linear(4608,128)
            # lin_5 = nn.linear(256,128)
            for weight in shadow_model.layer4[0].left[0].weight:
                x = weight.reshape(-1) #2304
                x = window(x,2304,128)
                layer4[cnt] = x
                nodes_feat.append(torch.reshape(x, (1, 128))[0])
                cnt += 1
            for weight in shadow_model.layer4[0].left[3].weight:
                x = weight.reshape(-1) #4608
                x = window(x,4608,128)
                layer4[cnt] = x
                nodes_feat.append(torch.reshape(x, (1, 128))[0])
                cnt += 1
            for weight in shadow_model.layer4[0].shortcut[0].weight:
                x = weight.reshape(-1) #256
                x = window(x,256,128)
                layer4[cnt] = x
                nodes_feat.append(torch.reshape(x, (1, 128))[0])
                cnt += 1
            for weight in shadow_model.layer4[1].left[0].weight:
                x = weight.reshape(-1) #4608
                x = window(x,4608,128)
                layer4[cnt] = x
                nodes_feat.append(torch.reshape(x, (1, 128))[0])
                cnt += 1
            fc_4 = {}
            for weight in shadow_model.layer4[1].left[3].weight:
                x = weight.reshape(-1) #4608
                x = window(x,4608,128)
                fc_4[cnt] = x
                nodes_feat.append(torch.reshape(x, (1, 128))[0])
                cnt += 1

            fc = {}
            # lin_6 = nn.linear(256,128)
            for weight in shadow_model.fc.weight:
                x = weight.reshape(-1) #512
                x = window(x,512,128)
                fc[cnt] = x
                nodes_feat.append(torch.reshape(x, (1, 128))[0])
                cnt += 1

            

            # layer3_4 = [] 
            # for src in layer3.keys():
            #     for dst in layer4.keys():
            #         layer3_4.append([src, dst])  
            layer4_fc = []
            for src in fc_4.keys():
                for dst in fc.keys():
                    layer4_fc.append([src, dst])    
   

            # get all nodes
            nodes_feat = torch.stack(nodes_feat)
            # print(nodes_feat.shape)
            # get all edges
            # all_edges = torch.tensor(ih_hh_l0 + hh_ih_l1 + ih_hh_l1 + hh_l1_att + att_out).t()
            all_edges = torch.tensor(#ih_l0_l1 +ih_l1_att +
                 layer4_fc).t()
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

   