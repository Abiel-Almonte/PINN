import math
import torch
from torch import nn
from typing import Tuple

SCALE: float= 10. #standard deviation to select B for RFFE

############################################################################################

## Random Fourier feature embedding (aka Gaussian encoding) -- mitgating spectral bias -> mapping inputs to higher dimensional feature space.
# gamma(_x)=  [cosine(2pi B(_x)), sine(2pi B(_x))]^T, B is selected independently from a normal distibution of N(0, sigma^2)
# sigma, scale factor, will need to carefully tuund in order to avoid catastrophic generalization

def sample_B(
    scale: float= SCALE,
    size: Tuple[int, int]= (None, None)
)-> torch.Tensor:
    
    ## torch.randn creates a random sample from the standard Gaussian distribution. 
    ## The std dev is changed via multiplication while the mean is changed by addition.

    return torch.randn(size)* scale

def rff_emb(
    _x: torch.Tensor,
    B: torch.Tensor
)-> torch.Tensor:

    _x_proj= torch.matmul((2* math.pi* _x), B) # matmul same as @ operator
    return torch.cat((torch.sin(_x_proj),  torch.cos(_x_proj)), dim=-1)

class RFFEmbeddings(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int
    )-> None:
        super().__init__()

        self.in_features= in_features
        self.out_features= out_features
        self.B= sample_B(size=(in_features, out_features))
    
    def forward(
        self,
        _x: torch.Tensor
    )-> torch.Tensor:
        
        return rff_emb(_x, self.B)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}'
        
############################################################################################

class MLP(nn.Module):
    def __init__(
        self,
        n_inputs: int= 2,
        neurons: int= 128,
        n_layers: int= 6,
        n_outputs: int= 1,
    )-> None:
        
        super().__init__()
        self._in= n_inputs
        self._neurons= neurons
        
        act_fn= nn.Tanh()

        self.input_layer= nn.Sequential(
            *[nn.Linear(n_inputs, n_inputs), act_fn]
        )

        self.embeddings= RFFEmbeddings(
            n_inputs, neurons
        )


        self.hidden_layers= nn.Sequential(
            *[nn.Sequential(
                *[nn.Linear(n_inputs, neurons), act_fn]
            ) for _ in range(n_layers-1)]
        )

        self.output_layer= nn.Linear(
            neurons, n_outputs
        )

    def forward(
        self,
        _x: torch.Tensor
    )-> torch.Tensor:

        _x= self.input_layer(_x)
        _x= self.embeddings(_x)
        _x= self.hidden_layers(_x)

        return self.output_layer(_x)