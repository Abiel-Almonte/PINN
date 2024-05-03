import torch
from functools import partial
from model import WaveEqNN, exact_solution
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt 

DEVICE='cuda'

def train_save_model(
    x: torch.Tensor,
    t: torch.Tensor
)-> WaveEqNN:

    model= WaveEqNN(x,t)
    model.train()
    
    return model

def plot(i, u_net):
    ax1.clear()
    ax2.clear()
    ax2.plot(x.cpu().numpy(),u_net(torch.hstack((x,torch.full(x.size(),t[i].item(), device= DEVICE)))).detach().cpu().numpy(), color= 'red')
    ax1.plot(x.cpu().numpy(),exact_solution(x,torch.full(x.size(),t[i].item(), device= DEVICE)).cpu().numpy())
    ax1.axis(ymin=-1.5, ymax=1.5)
    ax2.axis(ymin=-1.5, ymax=1.5)

if __name__ == '__main__':

    x= torch.rand(16392, device='cuda', dtype= torch.float32, requires_grad=True).unsqueeze(-1)
    t= torch.rand(16392, device='cuda', dtype= torch.float32, requires_grad=True).unsqueeze(-1)
    x.retain_grad()
    t.retain_grad()

    model= train_save_model(x, t)

    print("""
Best L1 Loss: {:0.6f}
""".format(model.best_loss))
    
    model.DNN.load_state_dict(model.best_model)
    x= torch.linspace(0,1, 100, device= DEVICE).unsqueeze(-1)
    t= torch.linspace(0,1, 100, device= DEVICE).unsqueeze(-1)
    fig, (ax1, ax2) =plt.subplots(1, 2)
    
    anim= FuncAnimation(fig, partial(plot, u_net= model.DNN), 100, interval= 10)
    anim.save('Wave1D.gif', dpi=200, writer='pillow')