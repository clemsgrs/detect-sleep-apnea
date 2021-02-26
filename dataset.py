import torch
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split


def binary_to_smooth(arr):
    y_smooth = np.zeros(len(arr))
    z_arr = np.array([0])
    arr_pad = np.hstack((np.hstack((z_arr,arr)),z_arr))
    Ends = np.where(((arr_pad[:-1]>arr_pad[1:])[1:]))[0]
    Begins = np.where(((arr_pad[:-1]<arr_pad[1:])[:-1]))[0]
    for k in range(len(Begins)):
        l = Ends[k]+1-Begins[k]
        midd = l//2
        y_smooth[Begins[k]:Ends[k]+1] = np.exp(-((np.arange(l)-midd*np.ones(l)))**2/((midd+pow(10,-5))**2))
    return(y_smooth)


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

  def __init__(self, data_df, target_df, signal_id=0, signal_freq=100, use_conv=True, smooth_y=True):

    self.data_df = data_df
    self.target_df = target_df
    self.signal_id = signal_id
    self.freq = signal_freq
    self.use_conv = use_conv
    self.smooth_y=smooth_y

  def __len__(self):
    return len(self.data_df)

  def __getitem__(self, idx):

    sample_index = self.data_df.loc[idx, 0]
    subject_index = self.data_df.loc[idx, 1]
    x = self.data_df.loc[idx, 2+9000*self.signal_id:2+9000*(self.signal_id+1)-1].values
    if self.use_conv:
      x = x.reshape(1, -1)
    else:
      x = x.reshape(-1, self.freq)
    y = self.target_df[self.target_df['ID'] == sample_index].values[0][1:]
    if(self.smooth_y):
        y = binary_to_smooth(y)
    return x, y


class OneChannelDataModule():

    def __init__(self, p):

        self.save_csv = p.save_csv
        self.signal_id = p.signal_id
        self.data_dir = p.data_dir
        self.data_file = p.data_file
        self.target_file = p.target_file
        self.use_conv = p.use_conv
        self.smooth_y = p.smooth_y

    def setup(self):

        tqdm.pandas()

        if Path(self.data_dir, 'train.csv').exists() and Path(self.data_dir, 'val.csv').exists():
            print(f'Loading train data from file...')
            train_df = pd.read_csv(Path(self.data_dir, 'train.csv'))
            print(f'...done.')
            print(f'Loading validation data from file...')
            val_df = pd.read_csv(Path(self.data_dir, 'val.csv'))
            print(f'...done.')
        else:
            train_df = pd.DataFrame(np.array(h5py.File(Path(self.data_dir, self.data_file), mode='r')['data']))
            train_df, val_df = train_test_split(train_df, test_size=0.3)
            if self.save_csv:
                train_df.to_csv(Path(self.data_dir, f'train.csv'), index=False)
                val_df.to_csv(Path(self.data_dir, f'val.csv'), index=False)

        target_df = pd.read_csv(Path(self.data_dir, self.target_file))

        # if Path(self.data_dir, f'test.csv').exists():
        #     print(f'Loading test slides from file...')
        #     test_df = pd.read_csv(Path(self.data_dir, f'test.csv'))
        #     print(f'...done.')
        # else:
        #     test_df = pd.read_csv(Path(self.data_dir, 'test', 'test_data.csv'))
        #     test_df = self.tile_dataframe(test_df, phase='test')
        #     test_df.to_csv(Path(self.data_dir, f'test.csv'), index=False)

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        self.train_dataset, self.val_dataset = (
            OneChannelDataset(train_df, target_df, signal_id=self.signal_id, use_conv=self.use_conv, smooth_y=self.smooth_y),
            OneChannelDataset(val_df, target_df, signal_id=self.signal_id, use_conv=self.use_conv, smooth_y=self.smooth_y)
        )
