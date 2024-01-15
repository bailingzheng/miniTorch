from torch.optim import Optimizer

__all__ = [
    'SGD'
]

class SGD(Optimizer):
    """Implements stochastic gradient descent (with momentum).

    The update can be written as
        G = G + weight_decay * P
        M = alpha * M + G
        P = P - lr * M

        where P, G, M denote the parameters, gradient, and momentum respectively.

    Moreover, the initial value of the momentum buffer is set to the gradient value at the first step. 

    Parameters
        params (iterable) - iterable of parameters to optimize or dicts defining parameter groups
        lr (float) - learning rate
        alpha (float, optional) - momentum factor (default: 0)
        weight_decay (float, optional) - weight decay (L2 penalty) (default: 0)

    """

    # torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, 
    #   nesterov=False, *, maximize=False, foreach=None, differentiable=False)
    def __init__(self, params, lr, alpha=0, weight_decay=0):
        defaults = dict(lr=lr, alpha=alpha, weight_decay=weight_decay)
        super(SGD, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['M'] = grad

                if state['step'] > 0:
                    if weight_decay != 0:
                        G += weight_decay * p.data
                    
                    state['M'] = alpha * state['M'] + grad
                    p.data += -lr * state['M']
                
                state['step'] += 1