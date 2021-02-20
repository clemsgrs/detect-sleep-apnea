import torch
import h5py
import pandas as pd

class SleepApneaDataset(torch.utils.data.Dataset):

  def __init__(self, data_path, csv_path, N_signals=8, signal_freq=100):

    self.dset = h5py.File(data_path, mode='r')['data']
    self.targets = pd.read_csv(csv_path)
    self.N = N_signals
    self.freq = signal_freq
  
  def __len__(self):
    return len(self.dset)
  
  def __getitem__(self, idx):
    
    sample_index = self.dset[idx, 0]
    subject_index = self.dset[idx, 1]
    x = self.dset[idx, 2:].reshape(-1, self.N)
    y = self.targets[self.targets['ID'] == sample_index].values[0][1:]

    return x, y

class OneChannelDataset(torch.utils.data.Dataset):

  def __init__(self, data_path, csv_path, signal_id=0, signal_freq=100):

    self.dset = h5py.File(data_path, mode='r')['data']
    self.targets = pd.read_csv(csv_path)
    self.signal_id = signal_id
    self.freq = signal_freq
  
  def __len__(self):
    return len(self.dset)
  
  def __getitem__(self, idx):
    
    sample_index = self.dset[idx, 0]
    subject_index = self.dset[idx, 1]
    x = self.dset[idx, 2+9000*self.signal_id:2+9000*(self.signal_id+1)]
    x = x.reshape(-1, self.freq)
    y = self.targets[self.targets['ID'] == sample_index].values[0][1:]

    return x, y