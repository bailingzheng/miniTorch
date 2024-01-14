from torch.optim import Optimizer

__all__ = [
    'RMSprop'
]

class RMSprop(Optimizer):
    """Implements RMSprop algorithm.

    For further details regarding the algorithm we refer to lecture notes by G. Hinton. 
    (https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

    The update can be written as
        V = alpha * V + (1 - alpha) * G**2
        P = P - lr * G / (V**0.5 + eps)

        where P, G, V denote the parameters, gradient, and velocity respectively.

    Parameters
        params (iterable) - iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional) - learning rate (default: 1e-2)
        alpha (float, optional) - smoothing constant (default: 0.99)
        eps (float, optional) - term added to the denominator to improve numerical stability (default: 1e-8)

    """

    # torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, 
    #   centered=False, foreach=None, maximize=False, differentiable=False)
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-08):
        defaults = dict(lr=lr, alpha=alpha, eps=eps)
        super(RMSprop, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['V'] = (1 - group['alpha']) * grad**2

                if state['step'] > 0:
                    state['V'] = group['alpha'] * state['V'] + (1 - group['alpha']) * grad**2
                    p.data += -group['lr'] * grad / (state['V']**0.5 + group['eps'])
                    
                state['step'] += 1
