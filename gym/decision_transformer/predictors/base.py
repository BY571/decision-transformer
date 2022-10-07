import torch
from torch import nn, Tensor


class BasePredictor(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_size, transformer, max_len):
        super(BasePredictor, self).__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.transformer = transformer
        
    def forward(self, ):
        raise NotImplementedError
    
    def get_action(self, ):
        raise NotImplementedError

    

