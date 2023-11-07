import gc
import numpy as np
import pytorch_lightning as L
import torch
from torch import nn
from pytorch_lightning import callbacks
from torch_geometric.data.lightning import LightningDataset
# from torch_geometric.data import lightning
from torch_geometric.nn.models import GIN
from torch_geometric.utils import train_test_split_edges
from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score
from models import ModelDataset_cifar
from models.gin_1 import GINModule
from torch_geometric.loader import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GAE,VGAE
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def epoch_meta_train_oc(meta_model, basic_model, optimizer, dataset):
    scores = []
    cum_loss = 0.0
    for i in range(len(dataset)):
        # data = train_test_split_edges(dataset[i])
        # x = data.x.to(device)
        # # print("这是y",dataset[i].y.item())
        # train_pos_edge_index = data.train_pos_edge_index.to(device)
        # out = basic_model(x,train_pos_edge_index)[1]
        out = dataset[i].x.to(device)

            # print(out.shape[0])
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

def epoch_meta_eval_oc(meta_model, basic_model, dataset,  threshold=0.0):
    preds = []
    labs = []
    for i in range(len(dataset)):
        # data = train_test_split_edges(dataset[i])
        # x = data.x.to(device)
        # train_pos_edge_index = data.train_pos_edge_index.to(device)
        # out = basic_model(x,train_pos_edge_index)[1]
        out = dataset[i].x.to(device)
        # print("这是out",out)
        score = meta_model.forward(out)

        preds.append(score.item())
        labs.append(dataset[i].y.item())

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


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        # in_channels 是特征数量, out_channels * 2 是因为我们有两个GCNConv, 最后我们得到embedding大小的向量
        # cached 因为我们只有一张图
        self.conv1 = GCNConv(in_channels, 4 * out_channels, cached=True) # 缓存仅用于转导学习
        self.conv2 = GCNConv(4 * out_channels,2* out_channels, cached=True) # cached only for transductive learning
        self.conv3 = GCNConv(2 * out_channels, out_channels, cached=True)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.conv3(x, edge_index)
    
class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # cached only for transductive learning
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
    



class MetaClassifierOC(nn.Module):
    def __init__(self, input_size=64, class_num=9, N_in=115, gpu=True):
        super(MetaClassifierOC, self).__init__()
        self.N_in = N_in
        self.N_h = 20
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


if __name__ == "__main__":
    # Free cuda memory and clear unused variables
    
    torch.cuda.empty_cache()
    gc.collect()
    GPU = True
    N_REPEAT = 3
    N_EPOCH = 10
    AUCs = []
    ACCs = []
    # devices = torch.cuda.device_count()
    # torch.cuda.memory_summary(device=devices, abbreviated=False)

    # Load Dataset
    # data_dir = "/home/ubuntu/date/hdd4/shadow_model_ckpt/mnist/models0"
    data_dir = "/home/jianghaoyu/cifar10_benign"
    test_dir = "/home/jianghaoyu/cifar10_outliers"
    dataset = ModelDataset_cifar(data_dir=data_dir)
    test_dataset = ModelDataset_cifar(data_dir=test_dir)
    # print(dataset.num_features)
    dataset = dataset.shuffle()
    test_dataset= test_dataset.shuffle()
    test_dataset = test_dataset
    # test_dataset = dataset[1*len(dataset) // 10:2 * len(dataset) // 10]
    # val_dataset = dataset[1*len(dataset) // 10:2 * len(dataset) // 10]
    train_dataset = dataset[1 * len(dataset) // 10:10 * len(dataset) // 10]
    print("train set: ", len(train_dataset))
    # print("valid set: ", len(val_dataset))
    print("test set: ", len(test_dataset))
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    ocmodel = MetaClassifierOC()
    # model = GAE(GCNEncoder(262656, 128))
    model = VGAE(VariationalGCNEncoder(262656,64))
    #通过 load_state_dict 函数加载参数，torch.load() 函数中重要的一步是反序列化。
    model.load_state_dict(torch.load("GAEmodel_cifar.pt"))
    model = model.to(device)
    ocmodel = ocmodel.to(device)

    for i in range(N_REPEAT):
        optimizer = torch.optim.Adam(ocmodel.parameters(), lr=1e-3)
        for _ in tqdm(range(N_EPOCH)):
            epoch_meta_train_oc(ocmodel,model,optimizer,train_dataset)
        test_info = epoch_meta_eval_oc(ocmodel,model,test_dataset,threshold='half')
        print("\tTest AUC:", test_info[0])
        print("\tTest ACC:", test_info[1])
        ACCs.append(test_info[1])
        AUCs.append(test_info[0])
    AUC_mean = sum(AUCs) / len(AUCs)
    ACC_mean = sum(ACCs) / len(ACCs)
    print("Average detection AUC on %d one-class classifier: %.4f" % (N_REPEAT, AUC_mean))
    print("Average detection ACC on %d one-class classifier: %.4f" % (N_REPEAT, ACC_mean))





    
 
    



