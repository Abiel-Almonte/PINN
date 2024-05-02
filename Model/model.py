from typing import Dict
from collections import OrderedDict
from Architecture import MLP
import torch
from torch import(
    autograd, nn, optim,
    sin, cos, 
    pi,
)

DOMAIN=(0,1)
DEVICE='cuda'
ETA= 5e-3 #lr
ALPHA: torch.float32= 0.3
C2: torch.float32= 4

def exact_solution( 
    x: torch.Tensor, 
    t: torch.Tensor
)-> torch.Tensor:
    
    return sin(pi*x)*cos(2*pi*t) \
        + 0.5*sin(4*pi*x)*cos(8*pi*t)

class WaveEqNN:
    def __init__(
        self,
        size: int
    )-> None:
        
        self.size= size

        self.null= torch.zeros((size, 1), device= DEVICE, requires_grad=True)
        self.ones= torch.ones((size, 1), device= DEVICE, requires_grad=True)

        self.DNN= MLP().to(DEVICE)
        self.optimizer= optim.Adam(self.DNN.parameters(),lr= ETA)
        self.mse= nn.MSELoss()
        self.lambdas= {'lambda_ic': 0, 'lambda_bc': 0, 'lambda_de': 0}
        self.loss= 0
        self.iter= 0

    def compute_result(
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
        res['u_xx']= autograd.grad(u_x, x, self.ones, create_graph= True)[0]
        res['u_tt']= autograd.grad(u_t, t, self.ones, create_graph= True)[0]

        return res

    def compute_loss(self):
        self.DNN.zero_grad()
        self.optimizer.zero_grad()

        coordinates= torch.linspace(DOMAIN[0], DOMAIN[1], self.size, device= 'cuda', requires_grad= True)
        idx_x, idx_t= torch.randperm(coordinates.nelement()), torch.randperm(coordinates.nelement())
        x, t= coordinates[idx_x], coordinates[idx_t]
        x, t= x.unsqueeze(-1), t.unsqueeze(-1)

        res= self.compute_result(x, t)
        g= exact_solution(x, self.null)

        ic_loss= self.mse(res['ic'], g)
        bc_loss= self.mse(res['bound1'], res['bound2'])
        de_loss= self.mse(res['u_tt'], C2*res['u_xx'])
        
        de_loss.backward(retain_graph=True)
        de_loss_l2norm= torch.cat([p.grad.flatten() for p in self.DNN.parameters() if p.grad is not None]).pow(2).sum().sqrt()
        self.DNN.zero_grad()

        bc_loss.backward(retain_graph=True)
        bc_loss_l2norm= torch.cat([p.grad.flatten() for p in self.DNN.parameters()]).pow(2).sum().sqrt()
        self.DNN.zero_grad()

        ic_loss.backward(retain_graph=True)
        ic_loss_l2norm= torch.cat([p.grad.flatten() for p in self.DNN.parameters()]).pow(2).sum().sqrt()
        self.DNN.zero_grad()

        l2_norm=  ic_loss_l2norm + bc_loss_l2norm + de_loss_l2norm

        lambda_new_ast= (
            lambda old_lambda, new_lambda: old_lambda*ALPHA + (1-ALPHA)*new_lambda
        )
        
        self.lambdas['lambda_ic']= lambda_new_ast(self.lambdas['lambda_ic'], l2_norm/ic_loss_l2norm)
        self.lambdas['lambda_bc']= lambda_new_ast(self.lambdas['lambda_bc'], l2_norm/bc_loss_l2norm)
        self.lambdas['lambda_de']= lambda_new_ast(self.lambdas['lambda_de'], l2_norm/de_loss_l2norm)

        self.loss=  self.lambdas['lambda_ic']*ic_loss + self.lambdas['lambda_bc']*bc_loss + self.lambdas['lambda_de']*de_loss
        self.loss.backward(retain_graph=True)
        self.iter+=1

        if not self.iter % 100:
            print('Iteration: {:}, MSE Loss: {:0.6f}'.format(self.iter, self.loss))
            print(self.lambdas)

        return self.loss
    
    def train(self):
        
        self.DNN.train()
        self.optimizer.step(self.compute_loss)
