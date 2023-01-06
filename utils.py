import json
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd


transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(), transforms.Resize((224,224))])

def save_data(dicti,name):
    ''' SAVE DATA into json format,
        INPUTs: dicti: Diz file, name: desired path for the dicti's data type. '''
    jfile = open(name, "w")
    jfile = json.dump(dicti, jfile)
    
def load_data(name):
    ''' LOAD DATA from json format,
        INPUTs: name: path where dicti json is located. 
        OUTPUT: dicti: Dictionary file '''
    jfile = open(name, "r")
    dicti = json.load(jfile)
    return dicti
def save_model(checkpoint, path):
  # This function saves a pytorch model.
    torch.save(checkpoint, path)

def load_model(path,model, device):
  # This function loads a pytorch model from a path.
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state']) 
    
    return checkpoint

def get_history_images(prediction_time, H, D, resolution):
    ''' This function is needed to take a solar wind measure occurring in a prediction date, and retrieve all the data requested for the forecasting. 
        INPUTs: prediction_time: Solar wind density datetime, H: History value, D: Time between last image and the prediction time, resolution: Sampling time of the data
        OUTPUT: list of the sun images date to take. '''
    prediction_time = pd.to_datetime(prediction_time, format = '%Y-%m-%dT%H:%M:%S')
    
    delay = pd.Timedelta(D, 'd') + pd.Timedelta(3, 'm') # 3 minutes of difference, due to the different datasets' samplings.
    history = pd.Timedelta(H, 'd')
    

    right_limit = prediction_time - delay # Closest time to the prediction
    left_limit = right_limit - history

    times_of_interest = [str(left_limit)]
    current_time = left_limit

    while current_time != right_limit:

        current_time += resolution
        times_of_interest.append(str(current_time))

    
    return times_of_interest



class DataSet(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        
        return len(self.dataset)

    def __getitem__(self, idx):

        return self.dataset.iloc[idx, 0], self.dataset.iloc[idx, 1], self.dataset.iloc[idx, 2]

    