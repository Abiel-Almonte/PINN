from typing import Dict
from collections import OrderedDict
from architecture import MLP
import torch
from torch import(
    autograd, nn, optim,
    sin, cos, 
    pi,
)

DOMAIN=(0,1)
DEVICE='cuda'
ETA= 0.5 #lr
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
        x:torch.Tensor,
        t:torch.Tensor,
    )-> None:
        
        self.x= x
        self.t= t

        self.zeros= torch.zeros((x.shape[0], 1), device= DEVICE)
        self.ones= torch.ones((x.shape[0], 1), device= DEVICE)

        self.DNN= MLP().to(DEVICE)

        self.criterion= nn.L1Loss(reduction='mean')
        self.optimizer= optim.LBFGS(
            self.DNN.parameters(),
            lr= ETA, 
            tolerance_grad=1e-05,
            history_size=200,
            max_iter=200000,
            max_eval=50000,
            line_search_fn="strong_wolfe",
        )

        self.iter= 0
        self.best_model= None
        self.best_loss= float('inf')

    def compute_residuals(self)-> Dict[str, torch.Tensor]:
        
        v= torch.hstack((self.x,self.t))
        v_ic= torch.hstack((self.x, self.zeros))
        v_bound1= torch.hstack((self.zeros,self.t))
        v_bound2= torch.hstack((self.ones,self.t))

        res= OrderedDict()
        res['ic']= self.DNN(v_ic)
        res['bound1']= self.DNN(v_bound1)
        res['bound2']= self.DNN(v_bound2)
        res['u_net']= self.DNN(v)

        u_x= autograd.grad(res['u_net'], self.x, grad_outputs=torch.ones_like(res['u_net']), create_graph=True, retain_graph= True)[0]
        u_t= autograd.grad(res['u_net'], self.t, grad_outputs=torch.ones_like(res['u_net']), create_graph=True, retain_graph= True)[0]

        u_xx= autograd.grad(u_x, self.x, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph= True)[0]
        u_tt= autograd.grad(u_t, self.t, grad_outputs=torch.ones_like(u_t), create_graph=True, retain_graph= True)[0]

        res['f']= u_tt - C2*u_xx

        return res

    def closure(self):
        self.optimizer.zero_grad()

        res= self.compute_residuals()
        g= exact_solution(self.x, self.zeros)
        u= exact_solution(self.x, self.t)

        ic_loss= self.criterion(res['ic'], g)
        bc_loss= self.criterion(res['bound1'], res['bound2'])
        r_loss= self.criterion(res['f'], self.zeros)
        #u_loss= self.criterion(res['u_net'],u )

        loss=  r_loss+ ic_loss + bc_loss
        loss.backward()

        self.iter+=1

        if not self.iter % 100:
            print('Iteration: {:}, L1 Loss: {:0.6f}'.format(self.iter, loss))
        
        if self.best_loss> loss:
            self.best_loss= loss
            self.best_model= self.DNN.state_dict()

        return loss
    
    def train(self):
        
        self.DNN.train()
        self.optimizer.step(self.closure)