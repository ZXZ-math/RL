import torch as torch 
import torch.nn as nn
from collections import OrderedDict

import torch 
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_size:int, action_size:int, hidden_size:int=256,non_linear:nn.Module=nn.ReLU):
        """
        input: tuple[int]
            The input size of the image, of shape (channels, height, width)
        action_size: int
            The number of possible actions
        hidden_size: int
            The number of neurons in the hidden layer

        This is a seperate class because it may be useful for the bonus questions
        """
        super(MLP, self).__init__()
        #====== TODO: ======
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.output = nn.Linear(hidden_size,action_size)
        self.non_linear = non_linear()

    def forward(self, x:torch.Tensor)->torch.Tensor:
        #====== TODO: ======
        x = self.linear1(x)
        x = self.non_linear(x)
        x = self.output(x) 
        return x 

class Nature_Paper_Conv(nn.Module):
    """
    A class that defines a neural network with the following architecture:
    - 1 convolutional layer with 32 8x8 kernels with a stride of 4x4 w/ ReLU activation
    - 1 convolutional layer with 64 4x4 kernels with a stride of 2x2 w/ ReLU activation
    - 1 convolutional layer with 64 3x3 kernels with a stride of 1x1 w/ ReLU activation
    - 1 fully connected layer with 512 neurons and ReLU activation. 
    Based on 2015 paper 'Human-level control through deep reinforcement learning' by Mnih et al
    """
    def __init__(self, input_size:tuple[int], action_size:int,**kwargs):
        """
        input: tuple[int]
            The input size of the image, of shape (channels, height, width)
        action_size: int
            The number of possible actions
        **kwargs: dict
            additional kwargs to pass for stuff like dropout, etc if you would want to implement it
        """
        super(Nature_Paper_Conv, self).__init__()
        
       
        
        self.CNN = nn.Sequential(
            nn.Conv2d(in_channels=input_size[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
            )
        
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_size[1], 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_size[2], 8, 4), 4, 2), 3, 1)
        self.linear_input_size = convw * convh * 64
        
        self.MLP = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(in_features=self.linear_input_size, out_features=512)),
            ('relu4', nn.ReLU()),
            ('output', nn.Linear(in_features=512, out_features=action_size))
        ]))

       

    def forward(self, x:torch.Tensor)->torch.Tensor:
        #==== TODO: ====
        x = self.CNN(x)
        if x.ndim  == 3:
            x = nn.Flatten(0,-1)(x)
        if x.ndim == 4:
            x = nn.Flatten()(x)
        x = self.MLP(x)
        return x

        
        
    
    


