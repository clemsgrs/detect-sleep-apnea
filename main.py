import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

from models import create_model
from dataset import OneChannelDataModule
from utils import open_config_file, train_model, evaluate_model

params = open_config_file(config/default.json)
params.gpu_ids = [params.gpu_ids]
# set gpu ids
if len(params.gpu_ids) > 0:
    torch.cuda.set_device(params.gpu_ids[0])

args = vars(params)

print('------------ Options -------------')
for k, v in sorted(args.items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')

 
data_module = OneChannelDataModule(params)
data_module.setup()
train_dataset, val_dataset = data_module.train_dataset, data_module.val_dataset
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

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
