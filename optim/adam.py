import torch
from torch.optim import Optimizer

__all__ = [
    'Adam'
]

class Adam(Optimizer):
    """Implements Adam algorithm.

    For further details regarding the algorithm we refer to Adam: A Method for Stochastic Optimization.
    (https://arxiv.org/abs/1412.6980)

    The update can be written as
        G = G + weight_decay * P
        M = beta1 * M + (1 - beta1) * G
        V = beta2 * V + (1 - beta2) * G**2
        M = M / (1 - beta1**t)
        V = V / (1 - beta2**t)
        P = P - lr * M / (V**0.5 + eps)

        where P, G, M, V denote the parameters, gradient, momentum, and velocity respectively.

    Parameters
        params (iterable) - iterable of parameters to optimize or dicts defining parameter groups
        lr (float, Tensor, optional) - learning rate (default: 1e-3). 
        betas (Tuple[float, float], optional) - coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional) - term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional) - weight decay (L2 penalty) (default: 0)

    """

    # torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, *, 
    #   foreach=None, maximize=False, capturable=False, differentiable=False, fused=None)
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Adam, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['M'] = torch.zeros_like(grad)
                    state['V'] = torch.zeros_like(grad)

                if state['step'] > 0:
                    if weight_decay != 0:
                        grad += weight_decay * p.data
                    
                    state['M'] = beta1 * state['M'] + (1 - beta1) * grad
                    state['V'] = beta2 * state['V'] + (1 - beta2) * grad**2

                    M = state['M'] / (1 - beta1**state['step'])
                    V = state['V'] / (1 - beta2**state['step'])
                    p.data += -lr * M / (V**0.5 + eps)
                
                state['step'] += 1