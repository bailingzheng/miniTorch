import torch
from torch.optim import Optimizer

__all__ = [
    'RMSprop'
]

class RMSprop(Optimizer):
    """Implements RMSprop algorithm.

    For further details regarding the algorithm we refer to lecture notes by G. Hinton. 
    (https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

    The update can be written as
        G = G + weight_decay * P
        V = alpha * V + (1 - alpha) * G**2
        P = P - lr * G / (V**0.5 + eps)

        where P, G, V denote the parameters, gradient, and velocity respectively.

    Parameters
        params - iterable of parameters to optimize or dicts defining parameter groups
        lr - learning rate (default: 1e-2)
        alpha - smoothing constant (default: 0.99)
        eps - term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay - weight decay (L2 penalty) (default: 0)

    """

    # torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, 
    #   centered=False, foreach=None, maximize=False, differentiable=False)
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay)
        super(RMSprop, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['V'] = torch.zeros_like(grad)

                if state['step'] > 0:
                    if weight_decay != 0:
                        grad += weight_decay * p.data

                    state['V'] = alpha * state['V'] + (1 - alpha) * grad**2
                    p.data += -lr * grad / (state['V']**0.5 + eps)
                    
                state['step'] += 1
