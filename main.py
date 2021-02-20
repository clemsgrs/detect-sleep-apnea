import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from utils import open_config_file, train_model, evaluate_model
from models import create_model


data_path = 'data/X_train.h5'
label_path = 'data/y_train_tX9Br0C.csv'
train_dset = OneChannelDataset(data_path, label_path)
train_loader = torch.utils.data.DataLoader(train_dset, batch_size=16, shuffle=False)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='default.json', metavar='N', help='config file')
args = parser.parse_args()
params = open_config_file(args.config)

params.gpu_ids = [params.gpu_ids]
# set gpu ids
if len(params.gpu_ids) > 0:
    torch.cuda.set_device(params.gpu_ids[0])

args = vars(params)

print('------------ Options -------------')
for k, v in sorted(args.items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')

model = create_model(params)

for epoch in range(params.nepochs):
    
    epoch_start_time = time.time()
    
    train_loss = train_model(model, train_loader, optimizer, criterion, params):

    if epoch % params.eval_every == 0:

        valid_loss = evaluate_model(model, val_loader, criterion)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_model.pt')
    
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'End of epoch {epoch+1} / {params.nepochs+1} \t Time Taken:  {epoch_mins}m {epoch_secs}s')
