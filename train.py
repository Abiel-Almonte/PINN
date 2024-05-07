import torch
from model import WaveNN
from util import u
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

DEVICE='cuda'
A= 0.5
C= 2

def plot(i):

    t, x = np.mgrid[0:1.005:0.005, 0:1.005:0.005]
    t= t.flatten()[:, None]
    x= x.flatten()[:, None]
    v= np.hstack((t,x))
    v_tensor= torch.tensor(v, device=DEVICE).float()

    u_exact= u(v_tensor, A, C).detach().cpu().numpy()
    model.DNN.load_state_dict(model.evolution[i])
    u_pred= model(v_tensor)

    t= t.reshape(201, 201)
    x= x.reshape(201, 201)
    u_exact= u_exact.reshape(201,201)
    u_pred= u_pred.reshape(201,201)

    ax= axes[0]
    ax.clear()
    c= ax.pcolor(t, x, u_exact, cmap='RdBu', vmin=-1.5, vmax=1.5, shading='auto')
    ax.set(ylabel= '$x$', xlabel='$t$', title= 'Exact u(t, x)')
    if i==1: fig.colorbar(c, ax=ax)

    ax= axes[1]
    ax.clear()
    c= ax.pcolormesh(t, x, u_pred, cmap='RdBu', vmin=-1.5, vmax=1.5, shading='auto')
    ax.set(ylabel= '$x$', xlabel='$t$', title= f'Predicted u(t, x)')
    if i== 1: fig.colorbar(c, ax=ax)

    ax= axes[2]
    ax.clear()
    c= ax.pcolormesh(t, x, np.abs(u_exact - u_pred), cmap='RdBu', vmin=-1.5, vmax=1.5, shading='auto')
    ax.set(ylabel= '$x$', xlabel='$t$', title= 'Absolute Error $\Delta u(t, x)$')
    if i== 1: fig.colorbar(c, ax=ax)

    ax= axes[3]
    ax.clear()
    ax.plot(model.loss_log[0:i])
    ax.set(yscale= 'log', xlabel= 'steps $\cdot 10^2$', ylabel= '$\mathcal{L}$', title= 'L2 Loss')
    ax.annotate(f'step: {i*100}', xy= (0.5, 0.92))
    fig.tight_layout()

if __name__ == '__main__':

    model= WaveNN()
    model.train(50000, 128)
    torch.save(model.best_model, 'waveNN.pt')

    fig, axes =plt.subplots(1, 4, figsize=(15, 4)) 

    anim= FuncAnimation(fig, partial(plot), frames=500)
    anim.save('waveNN-Animation.gif', fps=50, dpi=200, writer='pillow')
    plt.tight_layout()
    plt.show()