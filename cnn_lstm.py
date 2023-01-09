import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from init import *


class CNN_feature_extractor(nn.Module):

    def __init__(self):
        super(CNN_feature_extractor, self).__init__()
        self.conv1 = nn.Conv2d(CNN_LSTM['firstConv'], CNN_LSTM['secondConv'], kernel_size, padding = 'same')
        self.conv2 = nn.Conv2d(CNN_LSTM['secondConv'], CNN_LSTM['thirdConv'], kernel_size, padding = 'same')
        self.conv3 = nn.Conv2d(CNN_LSTM['thirdConv'], CNN_LSTM['lastConv'], kernel_size, padding = 'same')
        self.flat = Flatten()

        self.FC1 = nn.Linear(CNN_LSTM['hidden'], CNN_LSTM['hidden_output'])

        self.pool = nn.MaxPool2d(kernel_size)
        self.drop = nn.Dropout()

    def forward(self, x):

        x = self.drop(self.pool(F.relu(self.conv1(x))))
        x = self.drop(self.pool(F.relu(self.conv2(x))))
        x = self.pool(F.relu(self.conv3(x)))

        x = self.flat(x)

        x = F.relu(self.FC1(x))

        return x


class LSTM_feature_extractor(nn.Module):

    def __init__(self):
        super(LSTM_feature_extractor, self).__init__()
        self.LSTM = nn.LSTM(CNN_LSTM['hidden_output'], CNN_LSTM['LSTM_output'], CNN_LSTM['n_layers_LSTM'], batch_first = batch_first, dropout = CNN_LSTM['drop'])

    def forward(self, x):

        o, (h,c) = self.LSTM(x)

        x = F.relu(o)

        return x


class ShallowNetwork(nn.Module):

    def __init__(self):
        super(ShallowNetwork, self).__init__()
        self.FC1 = nn.Linear(CNN_LSTM['LSTM_output'], 2)

    def forward(self, x):

        x = self.FC1(x)

        return x

class CNN_LSTMmodule(nn.Module):

    def __init__(self):
        super(CNN_LSTMmodule, self).__init__()
        self.CNN_net = CNN_feature_extractor()
        self.LSTM_net = LSTM_feature_extractor()
        self.decoder = ShallowNetwork()

    def forward(self, x):

        bs = x.shape[0]
        pre_alloc = torch.zeros((batch_size, 2), device = 'cuda' if torch.cuda.is_available() else 'cpu')
        for sample_idx in range(bs):

            cnn_out = self.CNN_net(x[sample_idx])

            lstm_out = self.LSTM_net(cnn_out)

            lstm_out = lstm_out[-1]

            prediction = self.decoder(lstm_out)

            pre_alloc[sample_idx] = prediction
        
        output = pre_alloc

        return output




class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)
