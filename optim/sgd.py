from torch.optim import Optimizer

__all__ = [
    'SGD'
]

class SGD(Optimizer):
    """Implements stochastic gradient descent (with momentum).

    The update can be written as
        M = alpha * M + G
        P = P - lr * M

        where P, G, M denote the parameters, gradient, and momentum respectively.

    Moreover, the initial value of the momentum buffer is set to the gradient value at the first step. 

    Parameters
        params (iterable) - iterable of parameters to optimize or dicts defining parameter groups
        lr (float) - learning rate
        alpha (float, optional) - momentum factor (default: 0)

    """

    # torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, 
    #   nesterov=False, *, maximize=False, foreach=None, differentiable=False)
    def __init__(self, params, lr, alpha=0):
        defaults = dict(lr=lr, alpha=alpha)
        super(SGD, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum'] = grad

                if state['step'] > 0:
                    state['momentum'] = group['alpha'] * state['momentum'] + grad
                    p.data += -group['lr'] * state['momentum']
                
                state['step'] += 1