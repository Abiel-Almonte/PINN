from turtle import forward
from typing import Tuple, Dict
import math
import torch
from torch import nn

SIGMA_RFFE: float= 10. #standard deviation to select B for RFFE
SIGMA_RWF: float= 0.1 #standard deviation for selecting s in RWF
MU: float= 0.5 #mean for selecting s in RWF
EPSILON: float= 1. #scalar for computing temporal weight
ALPHA: float= 0.9 #scalar for computing gloabl weights
ETA: float= 5e-3 #learning rate for gradient descent


## Exact solution to compare final results
def exact_solution( 
    x: float, 
    t: float
)-> float:
    
    u= math.sin(math.pi*x)*math.cos(2*math.pi*t)\
        + 0.5*math.sin(4*math.pi*x)*math.cos(8*math.pi*t)

    return u

############################################################################################

## Random Fourier feature embedding (aka Gaussian encoding) -- mitgating spectral bias -> mapping inputs to higher dimensional feature space.
# gamma(_x)=  [cosine(2pi B(_x)), sine(2pi B(_x))]^T, B is selected independently from a normal distibution of N(0, sigma^2)
# sigma, scale factor, will need to carefully tuund in order to avoid catastrophic generalization

def sample_B(
    sigma: float= SIGMA_RFFE,
    size: Tuple[int, int]= (None, None)
)-> torch.Tensor:
    
    ## torch.randn creates a random sample from the standard Gaussian distribution. 
    ## The std dev is changed via multiplication while the mean is changed by addition.

    return torch.randn(size)* sigma

def rff_emb(
    _x: torch.Tensor,
    B: torch.Tensor
)-> torch.Tensor:

    _x_proj= torch.matmul((2* math.pi* _x), B) # matmul same as @ operator
    return torch.cat((torch.sin(_x_proj),  torch.cos(_x_proj)), dim=-1)

class GuassianInputEmbeddings(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        neurons: int
    )-> None:
        super().__init__()

        self.B= sample_B(size=(n_inputs, neurons))
    
    def forward(
        self,
        _x: torch.Tensor
    )-> torch.Tensor:
        
        return rff_emb(_x, self.B)
        
############################################################################################

## Random weight factorization (RWF) --mitigates spectral bais, and alteres the loss landscape allowing neurons to learn from its own adaptive learning rate
#-> initializing conventional linear layers where, w= s*v => W= diag(s)*V, on a coordinate based MLP, accelerating and improving training.
#  https://arxiv.org/pdf/2210.01274

def sample_s(
    sigma: float= SIGMA_RWF,
    mu: float= MU,
    size: Tuple[int, int]= (None, None)
)-> torch.Tensor:
    
    s= torch.randn(size)*sigma + mu
    return torch.exp(s)

def factorized_weights(
    in_features: int,
    out_features: int,
    factory_kwargs: Dict[str]
)-> Tuple[torch.Tensor]:
    
    w = torch.nn.Parameter(
            torch.empty(
                (in_features, out_features), **factory_kwargs
            )
        )
        
    s= sample_s(size= (in_features, out_features))
    v= w/s

    return v,s 
    

class FactorizedLinear(nn.Module):
    in_features: int
    out_features: int
    w: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None
    ) -> None:

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias= bias

        self.v, self.s= factorized_weights(
            in_features,
            out_features,
            {'device': device, 'dtype': dtype}
        )

        if bias: self.bias= torch.nn.Parameter(
                    torch.empty(
                        (out_features), 
                        **{'device': device, 'dtype': dtype}
                    )
                )
        else: self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.v*self.s, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.v*self.s)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(input, self.v*self.s, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

############################################################################################


class MLP:
    def __init__(
        self,
        n_inputs: int= 2,
        neurons: int= 128,
        n_layers: int= 6,
        n_outputs: int= 1,
    )-> None:
        
        self._in= n_inputs
        self._neurons= neurons
        
        act_fn= nn.Tanh()

        self.input_layer= nn.Sequential(
            *[GuassianInputEmbeddings(n_inputs, neurons), act_fn]
        )

        self.hidden_layers= nn.Sequential(
            *[nn.Sequential(
                *[FactorizedLinear(n_inputs, neurons), act_fn]
            ) for _ in range(n_layers-1)]
        )

        self.output_layer= FactorizedLinear(
            neurons, n_outputs
        )

    def forward(
        self,
        _x: torch.Tensor
    )-> torch.Tensor:

        _x= self.input_layer(_x)
        _x= self.hidden_layers(_x)

        return self.output_layer(_x)