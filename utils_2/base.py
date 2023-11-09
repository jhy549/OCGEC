# from abc import ABC, abstractmethod
import torch
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
import torch.nn.functional as F
from pathlib import Path
from torchmetrics import Accuracy
import torch.nn.utils.prune as prune
from tqdm import tqdm
from copy import deepcopy
from utils_2.get_fm import get_fm_resnet18cifar
import numpy as np
import json
class PruningBase():
    def __init__(self, model, val_loader, p_val_loader, args, all_val_loader=None, all_p_val_loader=None) -> None:
        self.model = model
        self.org_model = deepcopy(model)
        self.val_loader = val_loader
        self.all_val_loader = all_val_loader
        self.all_p_val_loader = all_p_val_loader
        self.p_val_loader = p_val_loader
        self.device = args.device
        self.dataset_name = args.dataset
        self.model_name = args.model
        self.log_dir = args.log_dir
        self.num_classes = args.num_classes
        self.input_size = args.input_size
        self.logger = args.logger
        self.conv_layers = None
        self.c_scores = None
        self.p_scores = None
        self.backdoor_neurons = None
        self.fm_list = None
        self.p_fm_list = None
        self.fm_score = None
        self.idx_rank = None
        self.node_dict = None
        self.pruned_dir = self.log_dir/'pruned_models'
        self.gradients = None
        if not Path.exists(self.pruned_dir):
            Path.mkdir(self.pruned_dir, parents = True)

    def get_convs(self):
        if self.model_name == "resnet18_cifar":
            conv_layers, _, _ = get_fm_resnet18cifar(self.model, torch.rand(1,3,self.input_size[0],self.input_size[1]).to(self.device))
        self.conv_layers = conv_layers

    def get_conv_fm(self):
        # self.logger.info('----------- Computing Activation Maps --------------')
        avg_fm = dict()
        # length = len(self.val_loader.sampler)
        length = len(self.val_loader.dataset)
        # print(length)
        for batch_idx, data in enumerate(self.val_loader):
            x, y = data
            x=x.to(self.device)
            if self.model_name == "resnet18_cifar":
                conv_layers, fm_list, _ = get_fm_resnet18cifar(self.model, x)
            for key in fm_list.keys():
                if batch_idx == 0:
                    avg_fm[key] = fm_list[key] /length
                else:
                    avg_fm[key] += fm_list[key] /length
        self.conv_layers = conv_layers
        self.fm_list = deepcopy(avg_fm)

        avg_fm = dict()
        length = len(self.p_val_loader.dataset)
        # print(length)
        for batch_idx, data in enumerate(self.p_val_loader):
            x, y, y_org = data
            x=x.to(self.device)
            if self.model_name == "resnet18_cifar":
                conv_layers, fm_list, _ = get_fm_resnet18cifar(self.model, x)
            for key in fm_list.keys():
                if batch_idx == 0:
                    avg_fm[key] = fm_list[key] /length
                else:
                    avg_fm[key] += fm_list[key] /length
        self.p_fm_list = deepcopy(avg_fm) 

    # def get_grid_avg_fm(self):
    #     def grid_fm(fm_imgs):
    #         grid_img = make_grid(fm_imgs, nrow=16, padding=1, normalize=False, pad_value=1)
    #         image = ToPILImage()(grid_img)
    #         return image
    #     def get_color(fm):
    #         #negative results (activated by poison but not clean input) is set to red, positive ones is set to green
    #         neg_mask = fm<0
    #         pos_mask = fm>0
    #         fm_r = fm.clone()
    #         fm_g = fm.clone()
    #         fm_b = torch.zeros_like(fm)
    #         fm_r[neg_mask] *= -1
    #         fm_r[pos_mask] *= 0 
    #         fm_g[neg_mask] *=0
    #         fm_g[pos_mask] *=1
    #         fm = torch.concat((fm_r,fm_g, fm_b), 1)
    #         return fm
    #     fm_list = self.fm_list
    #     p_fm_list = self.p_fm_list
    #     # print(fm_list['layer4.1.conv2'][0])
    #     if fm_list is None:
    #         fm_list = self.get_conv_fm()
    #     if p_fm_list is None:
    #         p_fm_list = self.get_conv_fm(poison=True)
    #     idx_rank = self.idx_rank
    #     # self.logger.info('----------- Generate Activation Imgs {} Rank --------------'.format('without' if idx_rank is None else 'with'))
    #     clean_folder = self.log_dir/'fm/clean_activation'
    #     poison_folder = self.log_dir/'fm/poison_activation'
    #     contrast_folder = self.log_dir/'fm/contrast_activation'
    #     Path(clean_folder).mkdir(parents=True, exist_ok=True)
    #     Path(poison_folder).mkdir(parents=True, exist_ok=True)
    #     Path(contrast_folder).mkdir(parents=True, exist_ok=True)
    #     counter = 0
    #     for i in fm_list.keys():
    #         fm = fm_list[i].unsqueeze(1)
    #         p_fm = p_fm_list[i].unsqueeze(1)
    #         if not idx_rank is None:
    #             rank = idx_rank[i]
    #             fm = torch.index_select(fm, 0, rank)
    #             p_fm = torch.index_select(p_fm, 0, rank)
    #         ubound = max(torch.max(fm), torch.max(p_fm))
    #         lbound = min((torch.max(fm), torch.max(p_fm)))
    #         fm = (fm-lbound)/(ubound-lbound)
    #         p_fm =(p_fm-lbound)/(ubound-lbound)
    #         c_fm = get_color(torch.clip((fm-p_fm)*2, min=-1, max=1))

    #         fm_image = grid_fm(fm)
    #         p_fm_image = grid_fm(p_fm)
    #         c_fm_image = grid_fm(c_fm)
    #         fm_image.save(clean_folder/'{:0>2d}_{}.jpg'.format(counter,i), quality = 100)
    #         p_fm_image.save(poison_folder/'{:0>2d}_{}.jpg'.format(counter,i), quality = 100)
    #         c_fm_image.save(contrast_folder/'{:0>2d}_{}.jpg'.format(counter,i), quality = 100)
    #         counter+=1

    # def validate(self, model = None, compute_ca = True, compute_asr = True):
    #     criterion = torch.nn.CrossEntropyLoss().to(self.device)
    #     metric = Accuracy(task="multiclass", num_classes=self.num_classes).to(self.device)
    #     if model == None:
    #         model = self.model
    #     model.eval()
    #     ca = None
    #     asr = None
    #     closs = 0
    #     ploss = 0
    #     # total_correct = 0.
    #     with torch.no_grad():
    #         if compute_ca:
    #             for batch_idx, data in enumerate(self.all_val_loader):
    #                 x, y = data
    #                 x, y = x.to(self.device), y.to(self.device)
    #                 out = model(x)
    #                 closs += criterion(out, y).item()
    #                 metric.update(out, y)
    #                 # pred = out.data.max(1)[1]
    #                 # total_correct += pred.eq(y.data.view_as(pred)).sum()
    #             ca = metric.compute()
    #             # ca = float(total_correct/len(self.val_loader.sampler))
    #             # total_correct = 0.
    #             closs /= len(self.all_val_loader)
    #             metric.reset()
    #         if compute_asr:
    #             for batch_idx, data in enumerate(self.all_p_val_loader):
    #                 x, y, y_org = data
    #                 x, y = x.to(self.device), y.to(self.device)
    #                 out = model(x)
    #                 ploss += criterion(out, y).item()
    #                 metric.update(out, y)
    #                 pred = out.data.max(1)[1]
    #             asr = metric.compute()
    #             # asr = float(total_correct/len(self.p_val_loader.sampler))
    #             ploss /= len(self.all_p_val_loader)
    #     return ca, asr, closs, ploss
    
    # def p_test(self, model = None, compute_ca = True, compute_asr = True):
    #     criterion = torch.nn.CrossEntropyLoss().to(self.device)
    #     metric = Accuracy(task="multiclass", num_classes=self.num_classes).to(self.device)
    #     if model == None:
    #         model = self.model
    #     model.eval()
    #     ca = None
    #     asr = None
    #     closs = 0
    #     ploss = 0
    #     # total_correct = 0.
    #     with torch.no_grad():
    #         if compute_ca:
    #             for batch_idx, data in enumerate(self.val_loader):
    #                 x, y = data
    #                 x, y = x.to(self.device), y.to(self.device)
    #                 out = model(x)
    #                 closs += criterion(out, y).item()
    #                 metric.update(out, y)
    #                 # pred = out.data.max(1)[1]
    #                 # total_correct += pred.eq(y.data.view_as(pred)).sum()
    #             ca = metric.compute()
    #             # ca = float(total_correct/len(self.val_loader.sampler))
    #             # total_correct = 0.
    #             closs /= len(self.val_loader)
    #             metric.reset()
    #         if compute_asr:
    #             for batch_idx, data in enumerate(self.p_val_loader):
    #                 x, y, y_org = data
    #                 x, y = x.to(self.device), y.to(self.device)
    #                 out = model(x)
    #                 ploss += criterion(out, y).item()
    #                 metric.update(out, y)
    #                 pred = out.data.max(1)[1]
    #             asr = metric.compute()
    #             # asr = float(total_correct/len(self.p_val_loader.sampler))
    #             ploss /= len(self.p_val_loader)
    #     return ca, asr, closs, ploss
    
    def setup(self):
        self.get_conv_fm()
        fm_dict = {'fm_list':self.fm_list, 'p_fm_list':self.p_fm_list, 'fm_score': self.fm_score, 'idx_rank':self.idx_rank}
        # torch.save(fm_dict, self.log_dir/'fm_dict.pt')
        # self.get_grid_avg_fm()
    
    # def prune_each_layer(self, ratio_list, log = False):
    #     model = deepcopy(self.model)
    #     conv_layers, _, _ = get_fm_resnet18cifar(model, torch.rand(1,3,self.input_size[0],self.input_size[1]).to(self.device))
    #     for i, key in enumerate(conv_layers.keys()):
    #         layer = conv_layers[key]
    #         idx_rank = self.idx_rank[key]
    #         if isinstance(ratio_list, list):
    #             ratio = ratio_list[i]
    #         elif isinstance(ratio_list, dict):
    #             ratio = ratio_list[key]
    #         pruned_num = self.prune_with_ratio(key, layer, idx_rank, ratio, log=log)
    #     return model
    
    # def prune_one(self, name, idx, layer = None, validate = True, recover = False):
    #     # if validate:
    #         # self.logger.info('----------- Start Pruning layer {} With idx {} --------------'.format(name, idx))
    #         # self.logger.info('Layer Name \t Neuron Idx \t CleanACC \t PoisonACC \t CleanLoss \t PoisonLoss')
    #     if layer == None:
    #         layer = self.conv_layers[name]
    #     if not hasattr(layer,'weight_mask'):
    #         layer = prune.identity(layer, 'weight')
    #     mask = layer.weight_mask
    #     if recover:
    #         mask[idx] = 1.0
    #     else:
    #         mask[idx] = 0.0
    #     if validate:
    #         ca, asr, closs, ploss = self.validate()
    #         # self.logger.info('{}     \t {}     \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(name, idx, ca, asr, closs, ploss))
    #         return ca, asr, closs, ploss

    def graph_construction(self):
        def process_feature(features, normalize = True):
            target_size = (4, 4)
            size = features.size()[1:]
            strides = (size[0]//target_size[0], size[1]//target_size[1])
            if strides == (1,1):
                features = features.flatten(1)
            else:
                features = F.max_pool2d(features, kernel_size = strides, stride = strides).flatten(1)
            if normalize:
                features = F.normalize(features, p=2, dim=0)
            return features

        def conv_sub_graph(node_cur, n_channels, edge_list, start = None):
            if start == None:
                start = node_cur
            end = node_cur+n_channels+1
            node_list = []

            for i in range(n_channels):
                edge_list.append([start, node_cur + (i+1)])
                node_list.append(node_cur + (i+1))
                edge_list.append([node_cur + (i+1), end])

            return edge_list, node_list, end
        

        node_cur = 0
        edge_list = []
        node_dict = dict()
        layer_names = list(self.conv_layers.keys())
        out_channels = [layer.out_channels for layer in self.conv_layers.values()]

        features = process_feature(self.fm_list['input'])
        key = 'conv1'
        edge_list,node_list, node_cur = conv_sub_graph(node_cur,out_channels[0],edge_list)
        processed_feature = process_feature(self.fm_list[key])
        features = torch.cat((features, processed_feature, torch.mean(processed_feature, dim=0).unsqueeze(0)))
        node_dict[key] = node_list

        layers = ['layer1', 'layer2', 'layer3', 'layer4']
        for layer in layers:
            node_start = node_cur
            key = layer+'.0' + '.conv1'
            index = layer_names.index(key)
            edge_list,node_list,node_cur = conv_sub_graph(node_cur,out_channels[index],edge_list)
            processed_feature = process_feature(self.fm_list[key])
            features = torch.cat((features, processed_feature, torch.mean(processed_feature, dim=0).unsqueeze(0)))
            node_dict[key] = node_list

            key = layer+'.0' + '.conv2'
            index = layer_names.index(key)
            edge_list,node_list,node_cur = conv_sub_graph(node_cur,out_channels[index],edge_list)
            processed_feature = process_feature(self.fm_list[key])
            features = torch.cat((features, processed_feature, torch.mean(processed_feature, dim=0).unsqueeze(0)))
            node_dict[key] = node_list

            node_end = node_cur
            key = layer+'.0' + '.shortcut.0'
            if not key in layer_names:
                edge_list.append([node_start,node_end])
                features[node_end] += features[node_start]
            else:
                index = layer_names.index(key)
                edge_list,node_list,node_cur = conv_sub_graph(node_cur,out_channels[index],edge_list, node_start)
                processed_feature = process_feature(self.fm_list[key])
                features = torch.cat((features, processed_feature, torch.mean(processed_feature, dim=0).unsqueeze(0)))
                edge_list.append([node_cur,node_end])
                features[node_end] += features[node_cur]
                node_dict[key] = node_list
           
            node_start = node_end
            key = layer+'.1' + '.conv1'
            index = layer_names.index(key)
            edge_list,node_list,node_cur = conv_sub_graph(node_cur,out_channels[index],edge_list)
            processed_feature = process_feature(self.fm_list[key])
            features = torch.cat((features, processed_feature, torch.mean(processed_feature, dim=0).unsqueeze(0)))
            node_dict[key] = node_list


            key = layer+'.1' + '.conv2'
            index = layer_names.index(key)
            edge_list,node_list,node_cur = conv_sub_graph(node_cur,out_channels[index],edge_list)
            node_end = node_cur
            processed_feature = process_feature(self.fm_list[key])
            features = torch.cat((features, processed_feature, torch.mean(processed_feature, dim=0).unsqueeze(0)))
            node_dict[key] = node_list
            edge_list.append([node_start,node_end])
            features[node_end] += features[node_start]

            self.node_dict = node_dict

        # self.logger.info("node num: {}, edge num: {}, feature size: {}".format(node_end+1, len(edge_list), features.shape))
        return list(map(list, zip(*edge_list))), features.cpu().numpy(), node_end+1
    
    # def reset_model(self):
    #     self.model = self.org_model
    #     self.org_model = deepcopy(self.org_model)
    
    # def prune_weight(self, name, idx, validate = True):
    #     state_dict = self.model.state_dict()
    #     state_dict["{}.{}".format(name, 'weight')][idx] = 0.0
    #     self.model.load_state_dict(state_dict)
    #     if validate:
    #         ca, asr, closs, ploss = self.validate()
    #         self.logger.info('{}     \t {}     \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(name, idx, ca, asr, closs, ploss))
    #         return ca, asr, closs, ploss

    # def get_backdoor_neurons(self):
    #     #Pre-activation Distributions Expose Backdoor Neurons
    #     #use CE loss to evaluate backdoor neurons
    #     self.c_scores = dict()
    #     self.p_scores = dict()
    #     # self.backdoor_neurons = dict()
    #     for key in list(self.conv_layers.keys()):
    #         # print(key)
    #         layer = self.conv_layers[key]
    #         channel_num = layer.out_channels
    #         c_scores = []
    #         p_scores = []
    #         for i in range(channel_num):
    #             closs1, ploss1 = self.p_test(ca=True)
    #             self.prune_one(key, i, validate = False)
    #             closs2, ploss2 = self.p_test(ca=True)
    #             c_score = closs1 - closs2
    #             p_score = ploss1 - ploss2
    #             c_scores.append(c_score)
    #             p_scores.append(p_score)
    #             layer.weight_mask[i] = 1.
    #         self.c_scores[key] = c_scores
    #         self.p_scores[key] = p_scores
    #         self.save_neurons()
    #         # self.backdoor_neurons[key] = np.argsort(scores)[::-1][:int(channel_num*0.2)]
    
    # # def prune_by_backdoor_neurons(self):
    # #     for key in list(self.conv_layers.keys())[19:]:
    # #         layer = self.conv_layers[key]
    # #         for idx in self.backdoor_neurons[key]:
    # #             print(self.ce_scores[key][idx])
    # #             self.prune_one(key, idx, layer, validate= True)
    # #         if input() == 'reset':
    # #             self.reset_model()
    
    # def save_neurons(self):
    #     scores = {'c_scores': self.c_scores, 'p_scores': self.p_scores}
    #     with open(self.log_dir/'scores{}.json'.format(len(list(self.c_scores.keys()))), 'w') as f:
    #         json.dump(scores, f)
    
    # def get_gradients(self):
    #     criterion = torch.nn.CrossEntropyLoss().to(self.device)
    #     loss = 0
    #     model = self.model
    #     model.eval()
    #     for batch_idx, data in tqdm(enumerate(self.val_loader)):
    #         x, y = data
    #         x, y = x.to(self.device), y.to(self.device)
    #         out = model(x)
    #         loss = criterion(out, y).cuda()
    #         loss.backward()
    #     for key in self.conv_layers.keys():
    #         layer=self.conv_layers[key]
    #         grad = layer.weight.grad
            # print(key)
            # print(grad)
