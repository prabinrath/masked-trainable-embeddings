import torch
from torch.utils.data import Dataset, DataLoader
import lzma
import pickle
import glob
import os


class ReverseTrajDataset(Dataset):
    def __init__(self, demo_dir, required_data_keys=['joint_positions','gripper_open'], task_filter_key=None, device='cpu'):
        self.device = device
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
    
    def _move_to_device(self, demo):
        demo_th = {}
        for key in demo:
            if len(demo[key]) > 0:
                demo_th[key] = [torch.as_tensor(dat, device=self.device) for dat in demo[key]]
        return demo_th
        
    def __getitem__(self, index):
        with lzma.open(self.file_list[index], 'rb') as f:
            demo_pair = pickle.load(f)
            forward = self._move_to_device(demo_pair['forward'])
            backward = self._move_to_device(demo_pair['backward'])
        
        return forward, backward


# dataset = ReverseTrajDataset(demo_dir='dataset/open_close', device='cuda')
# data_loader = DataLoader(dataset, batch_size=4, shuffle=True)