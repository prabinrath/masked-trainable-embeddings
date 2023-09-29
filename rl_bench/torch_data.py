import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import copy
import pickle
import glob
import os
import random

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class ConfigMinMaxNormalization(object):
    '''MinMax Normalization --> [-1, 1]
       x = (x - min) / (max - min).
       x = x * 2 - 1
    '''

    def __init__(self, bounds):
        assert bounds.shape[1] == 2
        self._min = bounds[:,0]
        self._max = bounds[:,0]

    def validate_bounds(self, js):
        return np.all(js>self._min and js<self._max)

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        X = X * 2. - 1.
        return X

    def inverse_transform(self, X):
        X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X
    

class ReverseTrajDataset(Dataset):
    task_filter_map = {'box': 1, 'door': 2}

    def __init__(self, file_list,
                 required_data_keys=['joint_positions','gripper_open'], 
                 chunk_size=10,
                 norm_bound=None, 
                 device='cpu'):
        self.file_list = file_list
        self.device = device
        self.required_data_keys = required_data_keys
        self.chunk_size = chunk_size
        if norm_bound is not None:
            self.normalizer = ConfigMinMaxNormalization(norm_bound)
        else:
            self.normalizer = None
        self.len = len(self.file_list)

    def __len__(self):
        return self.len
        
    def __getitem__(self, index):
        with open(self.file_list[index], 'rb') as f:
            demo = pickle.load(f)
            valid_keys = []
            for key in demo:
                if len(demo[key]) > 0:
                    valid_keys.append(key)
                    # pad terminal observations
                    demo[key] += [np.copy(demo[key][-1]),]*self.chunk_size

            assert len(valid_keys) > 0
            assert all([req_key in valid_keys for req_key in self.required_data_keys])

            episode_len = len(demo[valid_keys[0]])-self.chunk_size
            start_ts = np.random.choice(episode_len)

            data_batch = []
            for key in self.required_data_keys:
                data = demo[key][start_ts:start_ts+self.chunk_size]
                if 'position' in key and self.normalizer is not None:
                    for i, js in enumerate(data):
                        if self.normalizer.validate_bounds(js):
                            data[i] = self.normalizer.transform(js)
                        else:
                            raise Exception('out of bound joint state in demonstration')
                data = torch.as_tensor(np.stack(data, axis=0), device=self.device)
                if 'rgb' in key:
                    data = data.float() / 255
                data_batch.append(data)
            
            traj_type = self.file_list[index].split('_')[-2]
            if traj_type == 'forward':
                latent = np.array([start_ts/episode_len, 1., 0.], dtype=np.float32)
            elif traj_type == 'backward':
                latent = np.array([start_ts/episode_len, 0., 1.], dtype=np.float32)
            data_batch.append(torch.as_tensor(latent))

        return data_batch

def load_data(dataset_dir, required_data_keys, task_filter_key, chunk_size, norm_bound, batch_size=4, train_split=0.8):
        file_list = []
        for file in glob.glob(os.path.join(dataset_dir, '*.pickle')):
            if task_filter_key is None:
                file_list.append(file)
            else:
                task_type_key = int(file.split('.')[-2].split('_')[-1])%10
                if task_type_key == task_filter_key:
                    file_list.append(file)
        random.shuffle(file_list)

        split_idx = int(len(file_list))*train_split
        train_dataset = ReverseTrajDataset(file_list=file_list[:split_idx],
                                        required_data_keys=required_data_keys,
                                        chunk_size=chunk_size,
                                        norm_bound=norm_bound,
                                        device='cuda')
        val_dataset = ReverseTrajDataset(file_list=file_list[split_idx:],
                                        required_data_keys=required_data_keys,
                                        chunk_size=chunk_size,
                                        norm_bound=norm_bound,
                                        device='cuda')

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
