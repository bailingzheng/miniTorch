
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

    """

    # torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, *, 
    #   foreach=None, maximize=False, capturable=False, differentiable=False, fused=None)
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(Adam, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum'] = (1 - beta1) * grad
                    state['velocity'] = (1 - beta2) * grad**2

                if state['step'] > 0:
                    state['momentum'] = beta1 * state['momentum'] + (1 - beta1) * grad
                    state['velocity'] = beta2 * state['velocity'] + (1 - beta2) * grad**2

                    M = state['momentum'] / (1 - beta1**state['step'])
                    V = state['velocity'] / (1 - beta2**state['step'])
                    p.data += -group['lr'] * M / (V**0.5 + group['eps'])
                
                state['step'] += 1