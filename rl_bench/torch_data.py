import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import copy
import pickle
import glob
import os

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

class ReverseTrajDataset(Dataset):
    def __init__(self, demo_dir, 
                 required_data_keys=['joint_positions','gripper_open'], 
                 task_filter_key=None, 
                 chunk_size=10, 
                 device='cpu'):
        self.device = device
        self.required_data_keys = required_data_keys
        self.chunk_size = chunk_size
        self.file_list = []
        for file in glob.glob(os.path.join(demo_dir, '*.pickle')):
            if task_filter_key is None:
                self.file_list.append(file)
            else:
                task_type_key = int(file.split('.')[-2].split('_')[-1])%10
                if task_type_key == task_filter_key:
                    self.file_list.append(file)
                    
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
                data = torch.as_tensor(np.stack(data, axis=0), device=self.device)
                data_batch.append(data)
            
            traj_type = self.file_list[index].split('_')[-2]
            if traj_type == 'forward':
                latent = np.array([start_ts/episode_len, 1., 0.], dtype=np.float32)
            elif traj_type == 'backward':
                latent = np.array([start_ts/episode_len, 0., 1.], dtype=np.float32)
            data_batch.append(torch.as_tensor(latent))

        return data_batch


dataset = ReverseTrajDataset(demo_dir='dataset/open_close', device='cuda')
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
for data in data_loader:
    print(data)