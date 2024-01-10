from torch.optim import Optimizer

__all__ = [
    'SGD'
]

class SGD(Optimizer):
    """Implements stochastic gradient descent (with momentum).

    Considering the specific case of Momentum, the update can be written as
        V_t+1 = momentum * V_t + G_t
        P_t+1 = P_t - lr * V_t+1

        where P, G, V denote the parameters, gradient, and velocity respectively.

    Moreover, the initial value of the momentum buffer is set to the gradient value at the first step. 

    Parameters
        params (iterable) - iterable of parameters to optimize or dicts defining parameter groups
        lr (float) - learning rate
        momentum (float, optional) - momentum factor (default: 0)

    """

    # torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, 
    #   nesterov=False, *, maximize=False, foreach=None, differentiable=False)
    def __init__(self, params, lr, momentum=0):
        defaults = dict(lr=lr, momentum=momentum)
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

                if state['step'] > 0:
                    state['velocity'] = group['momentum'] * state['velocity'] + grad
                else:
                    state['velocity'] = grad
                
                p.data += -group['lr'] * state['velocity']
                state['step'] += 1