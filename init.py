import pandas as pd
from utils import *

project_name = 'A CNN-LSTM framework for the solar wind density forecasting'


wb = True
H = 4  # day
D = 2  # day

resolution = pd.Timedelta(1, 'd')


device = 'cuda' if torch.cuda.is_available() else 'cpu'



#architecture hyperparameters
n_channels = 1
n_hidden_channels = 8
kernel_size = (3,3)
n_layers = 1
batch_first = True
bias = True

convNet = {'firstConv': 8,
            'secondConv': 16,
            'thirdConv' : 32,
            'kernel': kernel_size,
            'drop': 0.3}





# Training hyperparameters
train_split, val_split, test_split = 0.6, 0.2, 0.2

path = './models/'


shuffle_train = True
shuffle_val = False
shuffle_test = False
batch_size = 2

training_hp = {'lr': 1e-4,
               'wd': 1e-2,
              'Scheduler': 'NO',
              'History': H+1,
              'Delay': D,
              'Time_resolution': resolution,
              'epochs': 5}

