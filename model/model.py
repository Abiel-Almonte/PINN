from typing import Callable, Tuple, Dict
from functools import partial
from architecture import MLP
from util import UTIL_FUNCS
import numpy as np
import copy
import torch
from torch import(
    autograd, optim,
)

DEVICE='cuda'
GAMMA= 0.9
ETA= 1e-3

class WaveNN:
    def __init__(
        self,
        config= UTIL_FUNCS,
        a: float= 0.5,
        c: float= 2.
    )-> None:
        
        self.config= config

        self.c= torch.tensor(c, requires_grad=True, device=DEVICE)
        self.a= torch.tensor(a, requires_grad=True, device=DEVICE)

        self.DNN= MLP().to(DEVICE)

        self.optimizer= optim.Adam(self.DNN.parameters(), ETA)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, GAMMA)
        
        self.iter=0
        self.loss_log=[]
        self.evolution= []
        self.best_model= None
        self.best_loss= float('inf')

        v=np.random.rand(int(1e5), 2)
        self.std= torch.tensor(v.std(0), requires_grad=True, device=DEVICE).float()
        self.mean= torch.tensor(v.mean(0), requires_grad=True, device=DEVICE).float()
        
    def u_net(
        self,
        t:torch.Tensor, 
        x:torch.Tensor
    )-> torch.Tensor:
        
        return self.DNN(torch.hstack((t, x)))

    def u_t_net(
        self, 
        t:torch.Tensor, 
        x:torch.Tensor
    )-> torch.Tensor:

        u = self.u_net(t, x)
        return autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]

    def r_net(
        self,
        t: torch.Tensor,
        x: torch.Tensor
    )-> torch.Tensor:
        
        u = self.u_net(t, x)
        return self.config['Residual Compute']['res_net'](u, t, x, self.c)

    def create_batch(
        self,
        func: Callable,
        batch_size: int
    )->Tuple[torch.Tensor, torch.Tensor]:
        
        v, o = func(batch_size, self.a, self.c)
        v= (v-self.mean)/ self.std
        return v, o
    
    def closure(
        self,
        batch_size: int
    )-> torch.Tensor:
            v_ic, u_ic = self.create_batch(self.config['Data Samplers']['ic'], batch_size)
            v_bc1= self.create_batch(self.config['Data Samplers']['bc1'], batch_size)[0]
            v_bc2= self.create_batch(self.config['Data Samplers']['bc2'], batch_size)[0]
            v_res= self.create_batch(self.config['Data Samplers']['res'], batch_size)[0]
            
            u_bc1_pred = self.DNN(v_bc1)
            u_bc2_pred = self.DNN(v_bc2)
            u_ic_pred = self.DNN(v_ic)
            u_t_ic_pred = self.u_t_net(v_ic[:, 0:1], v_ic[:, 1:2])
            r_pred = self.r_net(v_res[:, 0:1], v_res[:, 1:2])
            
            ic_loss = torch.mean((u_ic - u_ic_pred)** 2) + torch.mean(u_t_ic_pred** 2)
            bc_loss= torch.mean(u_bc1_pred** 2) + torch.mean(u_bc2_pred** 2)
            r_loss = torch.mean(r_pred** 2)

            loss = r_loss + ic_loss + bc_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.iter+= 1

            if not self.iter % 1000:
                self.lr_scheduler.step()

            if not self.iter % 100:

                self.loss_log.append(loss)
                self.evolution.append(copy.deepcopy(self.DNN.state_dict()))
                print(f'Iteration: {self.iter}, L2 Loss: {loss:.2e}, Residual: {r_loss:.2e}, Initial Conditions: {ic_loss:.2e}, Boundary Conditions: {bc_loss:.2e}')
            
            if loss< self.best_loss:
                self.best_loss= loss
                self.best_model= copy.deepcopy(self.DNN.state_dict())
                
            return loss

    def train(
        self,
        steps:int, 
        batch_size:int
    )-> None:
        
        self.DNN.train()
        for _ in range(steps):
            self.closure(batch_size)
        
           
    def __call__(
        self,
        v:torch.Tensor
    )-> torch.Tensor:
        v= (v-self.mean)/ self.std
        t= v[:, 0:1]
        x= v[:, 1:2]

        self.DNN.eval()

        u = self.u_net(t, x)
        u = u.detach().cpu().numpy()
        return u
