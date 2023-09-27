import torch
import torch.nn as nn

class SCL(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_h=None
        self.encoder_f=None
        self.decoder=None
        self.linear=None
        
    def encoder_fn(self):
          self.linear = nn.Sequential(
            nn.Linear(256, 1),
            nn.Tanh(),
            nn.Linear(256,1),
            nn.Tanh(),
            nn.Linear(2,1)
        )
 
    def decoder_fn(self):
           
        self.decoder = nn.Sequential(
            nn.Linear(256,1),
            nn.Tanh(),
            nn.Linear(256,1),
            nn.Tanh(),
            nn.Linear(7,2)
        )   