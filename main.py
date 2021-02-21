import time
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

from models import create_model
from dataset import OneChannelDataModule
from utils import open_config_file, train_model, evaluate_model, epoch_time

params = open_config_file('config/default.json')
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
optimizer = optim.Adam(model.parameters())
model = model.cuda()
criterion = nn.BCELoss()
criterion = criterion.cuda()

best_valid_loss = float('inf')
for epoch in range(params.nepochs):
    
    start_time = time.time()
    train_loss = train_model(epoch+1, model, train_loader, optimizer, criterion, params)

    if epoch % params.eval_every == 0:
        valid_loss = evaluate_model(epoch+1, model, val_loader, criterion, params)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_model.pt')
    
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'End of epoch {epoch+1} / {params.nepochs} \t Val loss = {np.round(valid_loss,6)} \t Time Taken:  {epoch_mins}m {epoch_secs}s\n')