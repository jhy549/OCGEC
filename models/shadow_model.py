import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa

class Model0(nn.Module):
    def __init__(self):
        super(Model0, self).__init__()
        self.name = 'Model0'

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=0)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32*4*4, 512)
        self.output = nn.Linear(512, 10)

    def forward(self, x):
        B = x.size()[0]

        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = F.relu(self.fc(x.view(B,32*4*4)))
        x = self.output(x)

        return x

    def loss(self, pred, label):
        return F.cross_entropy(pred, label)


class Model5(nn.Module):
    def __init__(self):
        super(Model5, self).__init__()
        self.name = 'Model5'

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear = nn.Linear(8*8*8, 256)
        self.fc = nn.Linear(256, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        B = x.size()[0]

        x = F.relu(self.conv1(x))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.max_pool(F.relu(self.conv4(x)))
        x = F.relu(self.linear(x.view(B,8*8*8)))
        x = F.dropout(F.relu(self.fc(x)), 0.5, training=self.training)
        x = self.output(x)

        return x

    def loss(self, pred, label):
        return F.cross_entropy(pred, label)
    

class Model6(nn.Module):
    def __init__(self, gpu=False):
        super(Model6, self).__init__()
        self.gpu = gpu
        self.lstm = nn.LSTM(input_size=40, hidden_size=100, num_layers=2, batch_first=True)
        self.lstm_att = nn.Linear(100, 1)
        self.output = nn.Linear(100, 10)

        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()

        # Torch version of melspectrogram , equivalent to:
        # mel_f = librosa.feature.melspectrogram(x, sr=sample_rate, n_mels=40)
        # mel_feature = librosa.core.power_to_db(mel_f)
        window = torch.hann_window(2048)
        if self.gpu:
            window = window.cuda()
        stft = (torch.stft(x, n_fft=2048, window=window).norm(p=2,dim=-1))**2
        mel_basis = torch.FloatTensor(librosa.filters.mel(16000, 2048, n_mels=40))
        if self.gpu:
            mel_basis = mel_basis.cuda()
        mel_f = torch.matmul(mel_basis, stft)
        mel_feature = 10 * torch.log10(torch.clamp(mel_f, min=1e-10))

        feature = (mel_feature.transpose(-1,-2) + 50) / 50
        lstm_out, _ = self.lstm(feature)
        att_val = F.softmax(self.lstm_att(lstm_out).squeeze(2), dim=1)
        emb = (lstm_out * att_val.unsqueeze(2)).sum(1)
        score = self.output(emb)
        return (score)

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)
    
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class Model7(nn.Module):
    def __init__(self, num_classes=10,gpu=False):
        super(Model7, self).__init__()
        self.gpu = gpu
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # self.ResidualBlock = ResidualBlock()
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

        if gpu:
            self.cuda()

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)
    

class WordEmb:
    # Not an nn.Module so that it will not be saved and trained
    def __init__(self, gpu, emb_path):
        w2v_value = np.load(emb_path)
        self.embed = nn.Embedding(*w2v_value.shape)
        self.embed.weight.data = torch.FloatTensor(w2v_value)
        self.gpu = gpu
        if gpu:
            self.embed.cuda()

    def calc_emb(self, x):
        if self.gpu:
            x = x.cuda()
        return self.embed(x)


class Model8(nn.Module):
    def __init__(self, gpu=False, emb_path='/home/jianghaoyu/Meta-Nerual-Trojan-Detection/raw_data/rt_polarity/saved_emb.npy'):
        super(Model8, self).__init__()
        self.gpu = gpu

        self.embed_static = WordEmb(gpu, emb_path=emb_path)
        self.conv1_3 = nn.Conv2d(1, 100, (3, 300))
        self.conv1_4 = nn.Conv2d(1, 100, (4, 300))
        self.conv1_5 = nn.Conv2d(1, 100, (5, 300))
        self.output = nn.Linear(3*100, 1)

        if gpu:
            self.cuda()
        
    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        if self.gpu:
            x = x.cuda()

        x = self.embed_static.calc_emb(x).unsqueeze(1)
        score = self.emb_forward(x)
        return score

    def emb_forward(self, x):
        if self.gpu:
            x = x.cuda()

        x_3 = self.conv_and_pool(x, self.conv1_3)
        x_4 = self.conv_and_pool(x, self.conv1_4)
        x_5 = self.conv_and_pool(x, self.conv1_5)
        x = torch.cat((x_3,x_4,x_5), dim=1)
        x = F.dropout(x, 0.5, training=self.training)
        score = self.output(x).squeeze(1)
        return score

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.binary_cross_entropy_with_logits(pred, label.float())

    def emb_info(self):
        emb_matrix = self.embed_static.embed.weight.data
        emb_mean = emb_matrix.mean(0)
        emb_std = emb_matrix.std(0, unbiased=True)
        return emb_mean, emb_std





    

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(self.expansion*planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Model9(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=10, norm_layer=nn.BatchNorm2d,gpu=False):
        super(Model9, self).__init__()
        if norm_layer is None:
            self._norm_layer = nn.BatchNorm2d
        else:
            self._norm_layer = norm_layer
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = self._norm_layer(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        if gpu:
            self.cuda()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self._norm_layer))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)

