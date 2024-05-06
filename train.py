import torch
from functools import partial
from util import UTIL_FUNCS
from model import WaveNN
from matplotlib.animation import FuncAnimation
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np

DEVICE='cuda'

def plot(i, fig, ax1, ax2,ax3):
    v = torch.hstack((torch.full_like(x, t[i].item(), device=DEVICE),x))

    ax1.clear()
    ax2.clear()
    ax3.clear()

    u_exact= UTIL_FUNCS['General Solutions']['u'](v, model.a, model.c).detach().cpu().numpy()
    u_pred= model(v)

    ax1.plot(x.detach().cpu().numpy(), u_exact)
    ax2.plot(x.detach().cpu().numpy(), u_pred)
    ax3.plot(x.detach().cpu().numpy(), abs(u_exact- u_pred))

    ax1.set(ylim=(-1.5, 1.5), title= r'Exact $u(t,x)$', xlabel=r'$x$', ylabel=r'$u$', label=f'$t={t[i].item()}$')
    ax2.set(ylim=(-1.5, 1.5), title= r'Predicted $u(t,x)$', xlabel=r'$x$', label=f'$t={t[i].item()}$')
    ax3.set(ylim=(-1.5, 1.5),title= r'Absolute Error $u(t,x)$', xlabel=r'$x$', ylabel=r'$\Delta u$' )
    fig.tight_layout()


if __name__ == '__main__':

    model= WaveNN()
    model.train(100000, 128)
    model.DNN.load_state_dict(model.best_model)

    example = 200
    t = torch.linspace(0, 1, example, device=DEVICE)
    x = torch.linspace(0, 1, example, device=DEVICE)
    t, x = torch.meshgrid(t, x)
    v_star = torch.hstack((t.flatten()[:, None], x.flatten()[:, None]))
    u_star = UTIL_FUNCS['General Solutions']['u'](v_star, 0.5,2).detach().cpu().numpy()
    
    u_pred= model(v_star)
    v_star= v_star.detach().cpu().numpy()
    t= t.detach().cpu().numpy()
    x= x.detach().cpu().numpy()
    U_star = griddata(v_star, u_star.flatten(), (t, x), method='cubic')
    U_pred = griddata(v_star, u_pred.flatten(), (t, x), method='cubic')
    torch.save(model.best_model, 'waveNN.pt')

    fig, (ax1, ax2, ax3) =plt.subplots(1, 3) 
    ax1.pcolor(t, x, U_star, cmap='viridis')
    ax1.set(ylabel= '$x$', xlabel='$t$', title= 'Exact u(t, x)')
    plt.colorbar()

    ax2.pcolor(t, x, U_pred, cmap='viridis')
    ax2.set(ylabel= '$x$', xlabel='$t$', title= 'Predicted u(t, x)')
    plt.colorbar()

    ax3.pcolor(t, x, np.abs(U_star - U_pred), cmap='viridis')
    ax2.set(ylabel= '$x$', xlabel='$t$', title= 'Absolute Error $\Delta u(t, x)$')
    plt.colorbar()
    plt.show()

    fig, (ax1, ax2, ax3)= plt.subplots(1, 3)
    anim= FuncAnimation(fig, partial(plot, fig=fig, ax1=ax1, ax2=ax2, ax3=ax3), frames= example, interval=10)
    anim.save('Wave1d_Comparision.gif', fps= 10, dpi=200, writer='pillow')