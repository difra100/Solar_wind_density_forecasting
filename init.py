import pandas as pd
from utils import *
import torch

project_name = 'A CNN-LSTM framework for the solar wind density forecasting'


wb = True
H = 4  # day
D = 2  # day

resolution = pd.Timedelta(1, 'd')


device = 'cuda' if torch.cuda.is_available() else 'cpu'

mod = 'cnnlstm'

#architecture hyperparameters
n_channels = 1

kernel_size = (3,3)
n_layers = 5
batch_first = True
bias = True
n_hidden_channels = 8

convNet = {'firstConv': n_hidden_channels,
            'secondConv': 16,
            'thirdConv' : 32,
            'kernel': kernel_size,
            'drop': 0.5}


CNN_LSTM = {'firstConv': 1,
            'secondConv': 8,
            'thirdConv': 16,
            'lastConv': 32,
            'hidden': 2048,
            'hidden_output': 256,
            'LSTM_output': 768,
            'n_layers_LSTM': 5,
            'drop': 0.4}


# Training hyperparameters
train_split, val_split, test_split = 0.6, 0.2, 0.2

path = './models/'


shuffle_train = True
shuffle_val = False
shuffle_test = False
batch_size = 4

training_hp = {'lr': 1e-5,
               'wd': 1e-1,
              'Scheduler': 'NO',
              'History': H+1,
              'Delay': D,
              'Time_resolution': resolution,
              'epochs': 100,
              'ConvLSTM_drop': 0.5,
              'batch_size': batch_size,
              'firstConv': convNet['firstConv'],
              'secondConv': convNet['secondConv'],
              'thirdConv' : convNet['thirdConv'],
              'kernel': kernel_size,
              'drop': convNet['drop']}

training_hp2 = {'lr': 1e-5,
               'wd': 1e-1,
              'Scheduler': 'NO',
              'History': H+1,
              'Delay': D,
              'Time_resolution': resolution,
              'epochs': 100,
              'ConvLSTM_drop': 0.5,
              'batch_size': batch_size,
              'firstConv': CNN_LSTM['firstConv'],
              'secondConv': CNN_LSTM['secondConv'],
              'thirdConv' : CNN_LSTM['thirdConv'],
              'kernel': kernel_size,
              'drop': CNN_LSTM['drop'],
              'lastConv': CNN_LSTM['lastConv'],
              'hidden': CNN_LSTM['hidden'],
              'hidden_output': CNN_LSTM['hidden_output'],
              'LSTM_output': CNN_LSTM['LSTM_output'],
              'n_layers_LSTM': CNN_LSTM['n_layers_LSTM']
              }


if mod == 'conv':
    hyperparameters = training_hp
elif mod == 'cnnlstm':
    hyperparameters = training_hp2