from argument_parser_v1 import parser_args
from model_building import build_model
from train_eval_test_v1 import train_eval_test_v1
from utils_1 import seed_setting, get_summary_writer, record_configuration
from models import ModelDataset_audio
from torch_geometric.loader import DataLoader
import torch 
from tqdm import tqdm
from torch import nn
import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score
import torch.nn.functional as F

def epoch_meta_train_oc(meta_model, basic_model,pooler,optimizer, dataset,device):
    scores = []
    cum_loss = 0.0
    # for i in range(len(dataset)):
    for i, batch_g in enumerate(dataset):
        # data = train_test_split_edges(dataset[i])
        # batch_g  = batch_g.to(device)
        # # print("这是y",dataset[i].y.item())
        # train_pos_edge_index = data.train_pos_edge_index.to(device)
        # out = basic_model(x,train_pos_edge_index)[1]

        out = basic_model.embed(batch_g, batch_g.x).to(device)
        out = pooler(x=out, batch=batch_g.batch).to(device)

        # print(out.shape)
        score = meta_model.forward(out)
        scores.append(score.item())

        loss = meta_model.loss(score)
        # print('loss', loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cum_loss += loss.item()
        meta_model.update_r(scores)
    return cum_loss / len(dataset)

def epoch_meta_eval_oc(meta_model, basic_model, pooler,dataset,device,  threshold=0.0):
    preds = []
    labs = []
    # for i in range(len(dataset)):
    for i, batch_g in enumerate(dataset):

        # data = train_test_split_edges(dataset[i])
        # x = data.x.to(device)
        # train_pos_edge_index = data.train_pos_edge_index.to(device)
        # out = basic_model(x,train_pos_edge_index)[1]
        # out = dataset[i].x.to(device)
        out = basic_model.embed(batch_g, batch_g.x).to(device)
        out = pooler(x=out, batch=batch_g.batch)
        # print("这是out",out)
        score = meta_model.forward(out)

        preds.append(score.item())
        labs.append(batch_g.y.item())

    preds = np.array(preds)
    labs = np.array(labs)
    auc = roc_auc_score(labs, preds)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labs, preds)
    # plt.figure()
    # plt.title('ROC')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.plot(false_positive_rate, true_positive_rate, '--*b', label="ours")
    # plt.legend()
    # plt.savefig("roc.jpg")
    # plt.show()



    if threshold == 'half':
        threshold = np.median(preds)
    acc = ( (preds>threshold) == labs ).mean()
    return auc, acc


class MetaClassifierOC(nn.Module):
    def __init__(self, input_size=64, class_num=1, N_in=512, gpu=True):
        super(MetaClassifierOC, self).__init__()
        self.N_in = N_in
        self.N_h = 512
        self.v = 0.1
        self.input_size = input_size
        self.class_num = class_num

        # self.inp = nn.Parameter(torch.zeros(self.N_in, *input_size).normal_()*1e-3)
        self.fc = nn.Linear(self.N_in*self.class_num, self.N_h)
        self.w = nn.Parameter(torch.zeros(self.N_h).normal_()*1e-3)
        self.r = 1.0

        self.gpu = gpu
        if self.gpu:
            self.cuda()

    def forward(self, pred, ret_feature=False):
        emb = F.relu(self.fc(pred.view(self.N_in*self.class_num)))
        if ret_feature:
            return emb
        score = torch.dot(emb, self.w)
        # print('这是score',score)
        return score

    def loss(self, score):
        reg = (self.w**2).sum()/2
        for p in self.fc.parameters():
            reg = reg + (p**2).sum()/2
        hinge_loss = F.relu(self.r - score)
        loss = reg + hinge_loss / self.v - self.r
        # print("loss",loss)
        return loss

    def update_r(self, scores):
        self.r = np.percentile(scores, 100*self.v)
        return
   #self.r是原点到超平面的距离

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
    data_dir = "/home/jianghaoyu/audio_benign"
    test_dir = "/home/jianghaoyu/audio_outliers"
    dataset = ModelDataset_audio(data_dir=data_dir)
    test_dataset = ModelDataset_audio(data_dir=test_dir)
    # print(dataset.num_features)
    # dataset = dataset.shuffle()
    # test_dataset= test_dataset.shuffle()
    # test_dataset = test_dataset
    # test_dataset = dataset[1*len(dataset) // 10:2 * len(dataset) // 10]
    # val_dataset = dataset[1*len(dataset) // 10:2 * len(dataset) // 10]
    train_dataset = dataset[2 * len(dataset) // 10:3 * len(dataset) // 10]
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    print('dataloader build finish')
    # change the model input dimension according to the dataset information
    model_config['in_dim'] = 102
    model, optimizer, scheduler, pooler = build_model(model_config, train_test_config)
    model.load_state_dict(torch.load("GAEmodel_audio.pt"))
    print('model build finish')
    ocmodel = MetaClassifierOC()
    ocmodel = ocmodel.to(train_test_config['device'])
    AUCs = []
    ACCs = []
    N_REPEAT = 1
    for i in range(N_REPEAT):
        optimizer = torch.optim.Adam(ocmodel.parameters(), lr=1e-3)
        for _ in tqdm(range(2)):
            epoch_meta_train_oc(ocmodel,model,pooler,optimizer,train_loader,train_test_config['device'])
        test_info = epoch_meta_eval_oc(ocmodel,model,pooler,test_loader,train_test_config['device'],threshold='half')
        print("\tTest AUC:", test_info[0])
        print("\tTest ACC:", test_info[1])
        ACCs.append(test_info[1])
        AUCs.append(test_info[0])
    AUC_mean = sum(AUCs) / len(AUCs)
    ACC_mean = sum(ACCs) / len(ACCs)
    print("Average detection AUC on %d one-class classifier: %.4f" % (N_REPEAT, AUC_mean))
    print("Average detection ACC on %d one-class classifier: %.4f" % (N_REPEAT, ACC_mean))
    # test_f1 = train_eval_test_v1(model, optimizer, scheduler, pooler, train_loader, test_loader, summary_writer, train_test_config)
    # return test_f1


if __name__ == '__main__':
    args = parser_args()
    import warnings
    warnings.filterwarnings("ignore")
    main(args)