from typing import Tuple
from Architecture import MLP
import math
import torch
from torch import(
    autograd,
    nn,
    optim,
)

DEVICE='cuda'
ETA= 1e-3  #lr
ALPHA= 0.5 #momentum

def exact_solution( 
    x: float, 
    t: float
)-> float:
    
    u= torch.sin(torch.pi*x)*torch.cos(2*torch.pi*t)\
        + 0.5*torch.sin(4*torch.pi*x)*torch.cos(8*torch.pi*t)

    return u

class WaveEqNN:

    def __init__(
        self,
        steps: int
    )-> None:
        
        coordinates= torch.linspace(0, 1, steps, device='cuda', requires_grad= True)
        idx_x, idx_t= torch.randperm(coordinates.nelement()), torch.randperm(coordinates.nelement())
        x, t= coordinates.view(-1)[idx_x], coordinates.view(-1)[idx_t]

        self.x, self.t= x.view(-1,1), t.view(-1,1)
        self.null= torch.zeros((x.shape[0],1))

        self.model= MLP().to(DEVICE)

        self.optimizer= optim.Adam(
            self.model.parameters(),lr= ETA
        )

        self.mse= nn.MSELoss()
        self.loss= 0
        self.iter=0

    def compute_res_grads(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    )-> Tuple[torch.Tensor, ...]:
        
        v= torch.hstack((x,t))
        u= self.model(v)

        u_x=  autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_xx= autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

        u_t= autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_tt= autograd.grad(u_t, t, torch.ones_like(u_t), create_graph=True)[0]

        v_ic= torch.hstack((x, torch.zeros_like(t)))
        u_ic= self.model(v_ic)
        u_t_ic= autograd.grad(u_ic, t, torch.ones_like(u), create_graph=True)[0]

        v_bound1= torch.hstack((torch.zeros_like(x),t))
        v_bound2= torch.hstack((torch.ones_like(x),t))
        u_bound1= self.model(v_bound1)
        u_bound2= self.model(v_bound2)

        return u_xx, u_tt, u_bound1, u_bound2, u_ic, u_t_ic

    def compute_loss(self):

        u_xx_pred, u_tt_pred, u_bound1_pred, u_bound2_pred, u_ic_pred, u_t_ic_pred= self.compute_res_grads(self.x, self.t)
        g= exact_solution(self.x, 0)

        ic_loss= self.mse(u_ic_pred, g) + self.mse(u_t_ic_pred, self.null)
        bc_loss= self.mse(u_bound1_pred, u_bound2_pred)
        r_loss= self.mse(u_tt_pred - 4*u_xx_pred, self.null)


