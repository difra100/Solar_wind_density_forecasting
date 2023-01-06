import pandas as pd

wb = True
H = 4  # day
D = 2  # day

resolution = pd.Timedelta(1, 'd')






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



shuffle_train = True
shuffle_val = False
shuffle_test = False
batch_size = 2

