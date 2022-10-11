import torch
from torch import nn, Tensor

class RndPredictor(nn.Module):
    def __init__(self, obs_size, hidden_size, output_size=None) -> None:
        super().__init__()
        
        self.input_layer = nn.Linear(obs_size, hidden_size)
        self.hidden1 = nn.Linear(hidden_size, hidden_size)
        if output_size == None:
            self.output = nn.Linear(hidden_size, obs_size)
        else:
            self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, obs: Tensor)-> Tensor:
        x = torch.relu(self.input_layer(obs))
        x = torch.relu(self.hidden1(x))
        return self.output(x)
        