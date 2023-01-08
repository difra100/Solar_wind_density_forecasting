import pandas as pd
from utils import *
import torch

project_name = 'A CNN-LSTM framework for the solar wind density forecasting'


wb = True
H = 4  # day
D = 2  # day

resolution = pd.Timedelta(1, 'd')


device = 'cuda' if torch.cuda.is_available() else 'cpu'



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


