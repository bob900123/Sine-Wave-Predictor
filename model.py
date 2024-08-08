import torch.nn as nn
import torch

class Network(nn.Module):
    def __init__(self, n_input:int, n_hidden:int, n_out:int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(n_input, n_hidden)
        self.layer2 = nn.Linear(n_hidden, n_out)
    
    def forward(self, x):
        x = self.layer1(x)
        x = torch.sigmoid(x)
        return self.layer2(x)