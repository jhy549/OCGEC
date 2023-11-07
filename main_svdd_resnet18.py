from argument_parser_v1 import parser_args
from model_building import build_model
from train_eval_test_v1 import train_eval_test_v1
from utils_1 import seed_setting, get_summary_writer, record_configuration
from models import ModelDataset_resnet
from torch_geometric.loader import DataLoader
import torch 
from tqdm import tqdm
from torch import nn
import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score
import torch.nn.functional as F

def eval(net, c, dataloader, device,pooler):
    """Testing the Deep SVDD model"""

    scores = []
    labels = []
    net.eval()
    print('Testing...')
    with torch.no_grad():
        for i,data in tqdm(enumerate(dataloader),total=len(dataloader),leave=True):
            # x = data.x.reshape(-1).to(device)
            data = data.to(device)
            # x = x.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            out = net.embed(data, data.x).to(device)
            z = pooler(x=out, batch=data.batch)
            # z = net(data.x, data.edge_index, data.batch)
            score = torch.sum((z - c) ** 2, dim=-1)
            # print("这是score",score)
            # print("这是y",data.y)
            # print(score.shape)
            scores.append(score.unsqueeze(0).detach().cpu())
            labels.append(data.y.unsqueeze(0).cpu())
    labels, scores = torch.cat(labels).numpy(), torch.cat(scores).numpy()
    print('ROC AUC score: {:.2f}'.format(roc_auc_score(labels, scores)*100))
    return labels, scores


class TrainerDeepSVDD:
    def __init__(self ,device,epoch,model,dataloader,pooler):
        self.epoch = epoch
        # self.train_loader, self.test_loader = data
        self.device = device
        self.model = model
        self.pooler = pooler

    def set_c(self,dataloader, eps=0.1):
        """Initializing the center for the hypersphere"""
        # z_ = []
        # with torch.no_grad():
        #     for data in dataloader:
        #         # print(data)
        #         x = data.x.reshape(-1).to(self.device)

        #         z_.append(x.detach())
        # z_ = torch.cat(z_)
        # c = torch.mean(z_, dim=0)
        # c[(abs(c) < eps) & (c < 0)] = -eps
        # c[(abs(c) < eps) & (c > 0)] = eps
        c = eps
        return c
    
    def train(self,dataloader):
        # net = network(in_channels=102, hidden_channels=64, out_channels=2).to(self.device)
        net = self.model.to(self.device)
        c = self.set_c(dataloader=dataloader)   
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001,
                               weight_decay=0.5e-6)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                    milestones=[50], gamma=0.1)

        net.train()
        for epoch in range(self.epoch):
            total_loss = 0
            for i,data in tqdm(enumerate(dataloader),total=len(dataloader),leave=True):
                # x = data.x.reshape(-1).to(self.device)
                data = data.to(self.device)
                # print(data.is_cuda)
                # x = x.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                optimizer.zero_grad()
                out = net.embed(data, data.x).to(self.device)
                # print(out.is_cuda)
                z = self.pooler(x=out, batch=data.batch)
                # z = net(data.x, data.edge_index, data.batch)
                loss = torch.mean(torch.sum((z - c) ** 2, dim=-1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            scheduler.step()
            print('Training Deep SVDD... Epoch: {}, Loss: {:.3f}'.format(
                   epoch, total_loss/len(dataloader)))
        self.net = net
        self.c = c


def main(args):
    seed = args.seeds
    seed_setting(seed)
    log_filepath = args.log_filepath
    model_config = {
        'num_layers': args.num_layers,
        'in_dim': args.in_dim,
        'num_hidden': args.num_hidden,
        'num_heads': args.num_heads,
        'num_out_heads': args.num_out_heads,
        'activation': args.activation,
        'feat_drop': args.feat_drop,
        'attn_drop': args.attn_drop,
        'negative_slope': args.negative_slope,
        'encoder': args.encoder,
        'decoder': args.decoder,
        'replace_rate': args.replace_rate,
        'mask_rate': args.mask_rate,
        'drop_edge_rate': args.drop_edge_rate,
        'optimizer_name': args.optimizer_name,
        'loss_fn': args.loss_fn,
        'linear_prob': args.linear_prob,
        'alpha_l': args.alpha_l,
        'norm': args.norm,
        'scheduler': args.scheduler,
        'pooling': args.pooling,
        'deg4feat': args.deg4feat,
        'residual': args.residual,
        'concat_hidden': args.concat_hidden,
    }
    train_test_config = {
        'seeds': seed,
        'device': args.device,
        'max_epoch': args.max_epoch,
        'max_epoch_f': args.max_epoch_f,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'lr_f': args.lr_f,
        'weight_decay_f': args.weight_decay_f,
    }
    summary_writer, log_dir = get_summary_writer(log_filepath)
    data_dir = "/home/jianghaoyu/Meta-Nerual-Trojan-Detection/resnet/cifar10/benign"
    test_dir = "/home/jianghaoyu/Meta-Nerual-Trojan-Detection/resnet/cifar10/2/"
    dataset = ModelDataset_resnet(data_dir=data_dir)
    test_dataset = ModelDataset_resnet(data_dir=test_dir)
    # print(dataset.num_features)
    # dataset = dataset.shuffle()
    # test_dataset= test_dataset.shuffle()
    # test_dataset = test_dataset
    # test_dataset = dataset[1*len(dataset) // 10:2 * len(dataset) // 10]
    # val_dataset = dataset[1*len(dataset) // 10:2 * len(dataset) // 10]
    train_dataset = dataset[2 * len(dataset) // 40:3 * len(dataset) // 40]
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    print('dataloader build finish')
    # change the model input dimension according to the dataset information
    model_config['in_dim'] = 128
    model, optimizer, scheduler, pooler = build_model(model_config, train_test_config)
    model.load_state_dict(torch.load("GAEmodel_resnet.pt"))
    print('model build finish')
    deep_SVDD = TrainerDeepSVDD(train_test_config['device'],2,model=model,dataloader=train_loader,pooler=pooler)
    deep_SVDD.train(train_loader)
    torch.save(deep_SVDD, "deep_SVDD_resnet.pt")
    labels, scores = eval(deep_SVDD.net, deep_SVDD.c, test_loader,train_test_config['device'],pooler=pooler)




    # test_f1 = train_eval_test_v1(model, optimizer, scheduler, pooler, train_loader, test_loader, summary_writer, train_test_config)
    # return test_f1


if __name__ == '__main__':
    args = parser_args()
    import warnings
    warnings.filterwarnings("ignore")
    main(args)
    
    
    #python main_svdd_resnet18.py  --device cuda:2 --max_epoch 2 --num_hidden 64