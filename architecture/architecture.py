import torch
from torch import nn
from typing import Tuple

SCALE: torch.float32= 1 #standard deviation to select B for RFFE

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

    return torch.randn(size, device='cuda')* scale

def rff_emb(
    v: torch.Tensor,
    B: torch.Tensor
)-> torch.Tensor:
    
    v= v @ B # matmul 

    v_proj=  2* torch.pi* v
    return torch.cat((torch.sin(v_proj),  torch.cos(v_proj)), dim=-1)

class RFFEmbeddings(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int
    )-> None:
        super().__init__()

        self.in_features= in_features
        self.out_features= out_features
        self.B= sample_B(size=(in_features, out_features//2))

        self.register_buffer('b', self.B)
    
    def forward(
        self,
        v: torch.Tensor
    )-> torch.Tensor:
        return rff_emb(v, self.B)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}'
        
############################################################################################

class MLP(nn.Module):
    def __init__(
        self,
        n_inputs: int= 2,
        neurons: int= 256,
        n_layers: int= 5,
        n_outputs: int= 1,
    )-> None:
        
        super().__init__()
        act_fn= nn.Tanh()

        self.input_layer= nn.Sequential(
            *[nn.Linear(n_inputs, n_inputs),
              RFFEmbeddings(n_inputs, neurons), act_fn]
        )


        self.hidden_layers= nn.Sequential(
            *[nn.Sequential(
                *[nn.Linear(neurons, neurons), act_fn]
            ) for _ in range(n_layers-1)]
        )

        self.output_layer= nn.Linear(
            neurons, n_outputs
        )

    def forward(
        self,
        v: torch.Tensor
    )-> torch.Tensor:

        v= self.input_layer(v) # v-> gamma
        v= self.hidden_layers(v) # gamma-> h

        return self.output_layer(v) # h-> u