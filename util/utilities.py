from typing import Tuple
import torch
from torch.autograd import grad
from torch import(
    sin, cos,
    pi,
)
DEVICE= 'cuda'

#General Exact Form
def u(
    v:torch.Tensor,
    a:torch.Tensor,
    c:torch.Tensor
)-> torch.Tensor:
    
    t,x = v[:,0:1], v[:,1:2]
    return sin(pi*x)*cos(c*pi*t) + a*sin(2*c*pi*x)*cos(4*c*pi*t)

def u_t(
    v:torch.Tensor,
    a:torch.Tensor,
    c:torch.Tensor
)-> torch.Tensor:
    
    t, x= v[:,0:1], v[:,1:2]
    return -c*pi*sin(pi*x)*sin(c*pi*t) - a*4*c*pi*sin(2*c*pi*x)*sin(4*c*pi*t)

def u_tt(
    v:torch.Tensor,
    a:torch.Tensor,
    c:torch.Tensor
)-> torch.Tensor:
    
    t, x= v[:,0:1], v[:,1:2]
    return -(c*pi)**2*sin(pi*x)*cos(c*pi*t) - a*(4*c*pi)**2*sin(2*c*pi*x)*cos(4*c*pi*t)

def u_xx(
    v:torch.Tensor,
    a:torch.Tensor,
    c:torch.Tensor
)-> torch.Tensor:
    
    t,x = v[:,0:1], v[:,1:2]
    return -pi**2*sin(pi*x)*cos(c*pi*t) - a*(2*c*pi)**2*sin(2*c*pi*x)*cos(4*c*pi*t)

#Residuals
def residual(
    v:torch.Tensor,
    a:torch.Tensor,
    c:torch.Tensor
)-> torch.Tensor:
    
    return u_tt(v, a, c) - c**2*u_xx(v, a, c)

def resdiual_net(
    u:torch.Tensor, 
    t:torch.Tensor, 
    x:torch.Tensor, 
    c:torch.Tensor
)->torch.Tensor:
    
    u_t = grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_x = grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_tt = grad(u_t, t, grad_outputs=torch.ones_like(u_t), retain_graph=True, create_graph=True)[0]
    u_xx = grad(u_x, x, grad_outputs=torch.ones_like(u_x),retain_graph=True,create_graph=True)[0]  
    return u_tt - c**2*u_xx

#Data samplers for calculating loss
def residual_data(
    batch_size: int,
    a: torch.Tensor,
    c: torch.Tensor
)-> Tuple[torch.Tensor, None]:
    
    v= torch.rand((batch_size, 2), requires_grad= True, device=DEVICE)
    #o= residual(v, a, c)
    return v, None

def ic_data(
    batch_size: int,
    a: torch.Tensor,
    c: torch.Tensor
)-> Tuple[torch.Tensor, torch.Tensor]:
    
    v= torch.tensor([0., 1.],device=DEVICE, requires_grad=True)*torch.rand((batch_size, 2), requires_grad= True, device=DEVICE)
    o= u(v, a, c)
    return v, o

def bc1_data(
    batch_size: int,
    a: torch.Tensor,
    c: torch.Tensor
)-> Tuple[torch.Tensor, None]:
    
    v= torch.tensor([1., 0.], device=DEVICE, requires_grad=True)*torch.rand((batch_size, 2), requires_grad= True).to(DEVICE)
    #o= 0
    return v, None

def bc2_data(
    batch_size: int,
    a: torch.Tensor,
    c: torch.Tensor
)-> Tuple[torch.Tensor, None]:

    v=  torch.tensor([0., 1.], device=DEVICE, requires_grad=True) + torch.tensor([1, 0], device=DEVICE)*torch.rand((batch_size, 2), requires_grad= True, device=DEVICE)
    #o= 0
    return v, None

UTIL_FUNCS={ 'General Solutions': 
            {'u': u, 'u_t': u_t, 'u_tt': u_tt, 'u_xx': u_xx},
            'Data Samplers':
            {'res': residual_data, 'ic': ic_data, 'bc1': bc1_data, 'bc2': bc2_data},
            'Residual Compute':
            {'res': residual, 'res_net': resdiual_net}}
