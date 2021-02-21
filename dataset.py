import torch
import h5py
import pandas as pd
from sklearn.model_selection import train_test_split

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

  def __init__(self, df, signal_id=0, signal_freq=100):

    self.dset = df
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


class OneChannelDataModule():
    
    def __init__(self, params):

        self.save_csv = p.save_csv
        self.signal_id = p.signal_id
        self.data_dir = p.data_dir
    
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
            train_df = pd.DataFrame(np.array(h5py.File(data_path, mode='r')['data']))
            train_df, val_df = train_test_split(train_df, test_size=0.3)
            if self.save_csv:
                train_df.to_csv(Path(self.data_dir, f'train.csv'), index=False)
                val_df.to_csv(Path(self.data_dir, f'val.csv'), index=False)

        # if Path(self.data_dir, f'test.csv').exists():
        #     print(f'Loading test slides from file...')
        #     test_df = pd.read_csv(Path(self.data_dir, f'test.csv'))
        #     print(f'...done.')
        # else:
        #     test_df = pd.read_csv(Path(self.data_dir, 'test', 'test_data.csv'))
        #     test_df = self.tile_dataframe(test_df, phase='test')
        #     test_df.to_csv(Path(self.data_dir, f'test.csv'), index=False)

        train_df = train_df.reset_index()
        val_df = val_df.reset_index()
        self.train_dataset, self.val_dataset = (
            OneChannelDataset(train_df, self.signal_id),
            OneChannelDataset(val_df, self.signal_id)
        )