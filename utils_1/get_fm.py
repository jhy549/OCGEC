import torch
import torch.nn as nn
import torch.nn.functional as F

def get_fm_resnet18cifar(model, input, with_shortcut = True, relu = True, l2 = False):
    def get_fm_basicblock(module, input, conv_layers, fm_list, prefix_key, relu, l2):
        out = module.bn1(module.conv1(input))
        key = prefix_key+'.conv1'
        if l2:
            fm_list[key]=torch.pow(out,2).sum(0)
        elif relu:
            fm_list[key]=F.relu(out.sum(0))
        else:
            fm_list[key]=out.sum(0)
        conv_layers[key]= module.conv1
        out = F.relu(out)

        out = module.bn2(module.conv2(out))
        key = prefix_key+'.conv2'
        if l2:
            fm_list[key]=torch.pow(out,2).sum(0)
        if relu:
            fm_list[key]=F.relu(out.sum(0))
        else:
            fm_list[key]=out.sum(0)
        conv_layers[key]=module.conv2

        shortcut_out = module.shortcut(input)
        if len(module.shortcut)!=0 and with_shortcut:
            key = prefix_key+'.shortcut.0'
            if relu:
                fm_list[key]=F.relu(shortcut_out.sum(0))
            else:
                fm_list[key]=shortcut_out.sum(0)
            conv_layers[key]=module.shortcut[0]

        out = out + shortcut_out
        out = F.relu(out)
        return conv_layers, fm_list, out

    fm_list= dict()
    conv_layers = dict()
    model.eval()
    with torch.no_grad():
        key = 'input'
        if l2:
            fm_list[key]=(torch.pow(input.sum(1), 2).sum(0)/3).unsqueeze(0)
        else:
            fm_list[key]=(input.sum(0).sum(0)/3).unsqueeze(0)
        out = model.bn1(model.conv1(input))
        key = 'conv1'
        if l2:
            fm_list[key]=torch.pow(out, 2).sum(0)
        elif relu:
            fm_list[key]=F.relu(out.sum(0))
        else:
            fm_list[key]=out.sum(0)
        conv_layers[key] = model.conv1
        out = F.relu(out)
        layers = [model.layer1, model.layer2, model.layer3, model.layer4]
        for i, layer in enumerate(layers):
            for j, basicblock in enumerate(layer.children()):
                key='layer{}.{}'.format(i+1,j)
                conv_layers, fm_list, out = get_fm_basicblock(basicblock, out, conv_layers, fm_list, key, relu, l2)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = model.linear(out)
    return conv_layers, fm_list, out

# def get_grid_avg_fm(fm_list, p_fm_list, idx_rank=None):
#     def grid_fm(fm_imgs):
#         grid_img = make_grid(fm_imgs, nrow=16, padding=1, normalize=False, pad_value=1)
#         image = ToPILImage()(grid_img)
#         return image

#     def get_color(fm):
#         #negative results (activated by poison but not clean input) is set to red, positive ones is set to green
#         neg_mask = (fm<0)
#         pos_mask = (fm>0)
#         fm_r = fm.clone()
#         fm_g = fm.clone()
#         fm_b = torch.zeros_like(fm)
#         fm_r[neg_mask] *= -1
#         fm_r[pos_mask] *= 0 
#         fm_g[neg_mask] *=0
#         fm_g[pos_mask] *=1
#         fm = torch.concat((fm_r,fm_g, fm_b), 1)
#         return fm

#     clean_folder = BASE_DIR/'outputs/clean_activation'
#     poison_folder = BASE_DIR/'outputs/poison_activation'
#     contrast_folder = BASE_DIR/'outputs/contrast_activation'
#     Path(clean_folder).mkdir(parents=False, exist_ok=True)
#     Path(poison_folder).mkdir(parents=False, exist_ok=True)
#     Path(contrast_folder).mkdir(parents=False, exist_ok=True)
#     counter = 0
#     for i in fm_list.keys():
#         fm = fm_list[i].abs().mean(0).unsqueeze(1)
#         p_fm = p_fm_list[i].abs().mean(0).unsqueeze(1)
#         if idx_rank != None:
#             rank = idx_rank[i]
#             fm = torch.index_select(fm, 0, rank)
#             p_fm = torch.index_select(p_fm, 0, rank)
#         ubound = torch.max(fm)
#         fm/=ubound
#         p_fm/=ubound
#         c_fm = get_color(torch.clip((fm-p_fm)*2, min=-1, max=1))
#         p_fm = torch.clip(p_fm, min=0, max=1)

#         fm_image = grid_fm(fm)
#         p_fm_image = grid_fm(p_fm)
#         c_fm_image = grid_fm(c_fm)
#         fm_image.save(clean_folder/'{:0>2d}_{}.jpg'.format(counter,i), quality = 100)
#         p_fm_image.save(poison_folder/'{:0>2d}_{}.jpg'.format(counter,i), quality = 100)
#         c_fm_image.save(contrast_folder/'{:0>2d}_{}.jpg'.format(counter,i), quality = 100)
#         counter+=1

# def prune_with_ratio(layer, idx_rank, ratio):
#     if not isinstance(layer, nn.Conv2d):
#         raise ValueError("layer should be conv!")
#     layer = prune.identity(layer, 'weight')
#     mask = layer.weight_mask
#     prune_num = ratio*layer.out_channels
#     counter = 0
#     for idx in idx_rank:
#         if mask[idx].norm(p=1) > 1e-6:
#             mask[idx] = 0.0
#             counter += 1
#             if counter >= prune_num:
#                 break

# def get_conv_fmrank(model, dataloader, device):
#     for data in dataloader:
#         x, y = data
#         conv_layers, fm_list, _ = get_fm_resnet18cifar(model, x.to(device))
#     idx_rank = dict()
#     fm_scores = dict()
#     for key in fm_list.keys():
#         # print(conv_layers[i])
#         # print(feature_map.shape)
#         idx, score = get_rank(fm_list[key])
#         idx_rank[key]=idx
#         fm_scores[key]=score
#         # print(idx)
#         # print(score)
#         # if i in fm_before_shortcut.keys():
#         #     print(get_rank(F.relu(fm_before_shortcut[i])))
#     return conv_layers, fm_list, fm_scores, idx_rank

# def get_convs(model):
#     model_weights = []
#     conv_layers = []
#     model_children = list(model.children())
#     counter = 0
#     for i in range(len(model_children)):
#         if type(model_children[i]) == nn.Conv2d:
#             counter+=1
#             model_weights.append(model_children[i].weight)
#             conv_layers.append(model_children[i])

#         elif type(model_children[i]) == nn.Sequential:
#             for j in range(len(model_children[i])):
#                 for child in model_children[i][j].children():
#                     if type(child) == nn.Conv2d:
#                         counter+=1
#                         model_weights.append(child.weight)
#                         conv_layers.append(child)
#     print(f"Total convolution layers: {counter}")
#     return conv_layers, model_weights