from configs import BASE_DIR
import torch

def load_model(args):
    if args.model == 'resnet18_cifar':
        from models.resnet_cifar import resnet18
        model = resnet18()
    state_dict = torch.load(BASE_DIR/'weights/attack/badnet/ResNet18-ResNet-BadNets-target0-portion0.1-epoch80.tar')
    model.load_state_dict(state_dict['state_dict'])
    model.to(args.device)

    # model2 = resnet18()
    # state_dict2 = process_state_dict(torch.load('weights/resnet_cifar10.bin'))
    # model2.load_state_dict(state_dict2)
    return model#, model2

def process_state_dict(state_dict):
    for key in list(state_dict.keys()):
        if 'downsample' in key:
            state_dict.update({key.replace('downsample', 'shortcut'):state_dict.pop(key)})
        if 'fc' in key:
            state_dict.update({key.replace('fc', 'linear'):state_dict.pop(key)})
    return state_dict