from typing import Dict
from collections import OrderedDict
from Architecture import MLP
import torch
from torch import(
    autograd, nn, optim,
    sin, cos, 
    pi,
)

DEVICE='cuda'
DOMAIN=(0,1) # & range 
ETA= 1e-3  #lr
ALPHA= 0.5 #momentum

def exact_solution( 
    x: torch.Tensor, 
    t: torch.Tensor
)-> torch.Tensor:
    
    return sin(pi*x)*cos(2*pi*t) \
        + 0.5*sin(4*pi*x)*cos(8*pi*t)

class WaveEqNN:

    def __init__(
        self,
        steps: int
    )-> None:
        
        coordinates= torch.linspace(DOMAIN[0], DOMAIN[-1], steps, device= DEVICE, requires_grad= True)
        idx_x, idx_t= torch.randperm(coordinates.nelement()), torch.randperm(coordinates.nelement())
        x, t= coordinates[idx_x], coordinates[idx_t]
        t.requires_grad()
        x.requires_grad()
        self.x, self.t= x.unsqueeze(-1), t.unsqueeze(-1)
        self.null= torch.zeros_like(self.x, device= DEVICE, requires_grad=True)
        self.ones= torch.ones_like(self.x, device= DEVICE, requires_grad=True)

        self.DNN= MLP().to(DEVICE)
        self.optimizer= optim.Adam(self.DNN.parameters(),lr= ETA)
        self.mse= nn.MSELoss()
        self.loss= 0
        self.iter= 0

    def compute_res(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    )-> Dict[str, torch.Tensor]:
        
        v= torch.hstack((x,t))
        v_ic= torch.hstack((x, self.null))
        v_bound1= torch.hstack((self.null,t))
        v_bound2= torch.hstack((self.ones,t))

        res= OrderedDict()

        u= self.DNN(v)
        res['ic']= self.DNN(v_ic)
        res['bound1']= self.DNN(v_bound1)
        res['bound2']= self.DNN(v_bound2)

        u_x= autograd.grad(u, x, self.ones, create_graph= True)[0]
        u_t= autograd.grad(u, t, self.ones, create_graph= True)[0]
        res['u_xx']= autograd.grad(u_x, x, self.ones, create_graph= True, retain_graph= True)[0]
        res['u_tt']= autograd.grad(u_t, t, self.ones, create_graph= True, retain_graph= True)[0]

        return res

    def compute_loss(self):

        res= self.compute_res(self.x, self.t)
        g= exact_solution(self.x, self.null)

        ic_loss= self.mse(res['ic'], g)
        bc_loss= self.mse(res['bound1'], res['bound2'])
        de_loss= self.mse(res['u_tt']- 4*res['u_xx'], self.null)
        
        loss= ic_loss + bc_loss + de_loss
        loss.backward()
