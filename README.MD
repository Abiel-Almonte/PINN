## Physics Informed Neural Networks (PINN):
Employing Deep Neural Networks (DNNs) to solve time dependent systems defined by laws of physics.

More specifically, DNN's automatic differentiation is utilizied to differentiate with respect to input and model parameters, yielding a network that "respects" any principles from physical laws interwoven deep within a observed system often represented as a time dependent partial differential equation with the following form: 

$$ u_t  + {\cal N [u]}= 0, t\in [0,T],  x\in\Omega$$
With the following Inital and Boundary Condition:
$$ u(t,x)|_{t=0}= u(0,x)= g(x)$$ 
$${\cal B[u]}= 0$$

$u$ is the unknown hidden solution governed by the PDE, substituted with a neural network with the from $u_\theta(t,x)$, where $\theta$ are all the parameters of the linear combination the models the neural network.

The residual (error) of the PDE is determined with:
$$\cal R_\theta(t,x)= u_\theta(t_r, x_r) + \cal N[u_\theta](t_r, x_r)$$

The loss function (MSE) of the DNN is the following:

$$\cal L(\theta)= \cal L_{ic}(\theta) + \cal L_{bc}(\theta) + \cal L_{r}(\theta)$$

Initial Condition Loss:
$${\cal L_{ic}}(\theta)= \frac{1}{N_{ic}}\sum_{i=1}^{N_{ic}} \left|u_\theta(0, x^i_{ic}) - g(x^i_{ic})\right|^2$$

Boundary Condition Loss:
$${\cal L_{bc}}(\theta)= \frac{1}{N_{bc}}\sum_{i=1}^{N_{bc}} \left|{\cal B[u_\theta]}(t^i_{bc}, x^i_{bc})\right|^2$$

Residual:
$${\cal L_{r}}(\theta)= \frac{1}{N_{r}}\sum_{i=1}^{N_{r}} \left|{\cal R_\theta}(t^i_{r}, x^i_{r})\right|^2$$

Where all $t, x$ are sampled during optimization.

[More Information on the theory of PINNs](https://doi.org/10.1016/j.jcp.2018.10.045)

### One-Dimensional Wave Equation:
$$\frac{\partial^2u}{\partial t^2}= 4\nabla^2_u \implies \frac{\partial^2u}{\partial t^2} - 4 \frac{\partial^2u}{\partial x^2} = 0, x\in [0,1], t\in [0,1]$$

Inital Conditions:
$$u(0,x)= g(x)= sin(\pi x)+ \frac{1}{2}sin(4\pi x), \frac{\partial u}{\partial t}(0,x)= f(x)= 0 $$ 

Boundaries:
$$ u(t,0)= u(t,1)= h(t,x)= 0$$

Exact Solution:
$$u(x,t)= sin(\pi x)cos(2\pi t)+ \frac{1}{2}sin(4\pi x)cos(8\pi t)$$

Loss:
$${\cal L_{ic}}(\theta)= \frac{1}{N_{ic}}\sum_{i=1}^{N_{ic}} \left|u_\theta(0, x^i_{ic}) - g(x^i_{ic})\right|^2 + \left|\frac{\partial u_\theta}{\partial t}(0, x^i_{ic})\right|^2 $$

$${\cal L_{bc}}(\theta)= \frac{1}{N_{bc}}\sum_{i=1}^{N_{bc}} \left|h_\theta(t^i_{bc}, 0) - h_\theta(t^i_{bc}, 1)\right|^2$$

$${\cal L_{r}}(\theta)= \frac{1}{N_{r}}\sum_{i=1}^{N_{r}} \left|u_{\theta}(t^i_r, x^i_r) -4u_{\theta\theta}(t^i_r, x^i_r)\right|^2$$