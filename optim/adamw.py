from torch.optim import Optimizer

__all__ = [
    'AdamW'
]


class AdamW(Optimizer):
    """Implements AdamW algorithm.

    For further details regarding the algorithm we refer to:
    Decoupled Weight Decay Regularization (https://arxiv.org/abs/1711.05101).

    Parameters
        params (iterable) - iterable of parameters to optimize or dicts defining parameter groups
        lr (float, Tensor, optional) - learning rate (default: 1e-3).
        betas (Tuple[float, float], optional) - coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional) - term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional) - weight decay coefficient (default: 1e-2)

    """

    # torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False, *,
    #   maximize=False, foreach=None, capturable=False, differentiable=False, fused=None)
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)

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
                    state['M'] = (1 - beta1) * grad
                    state['V'] = (1 - beta2) * grad**2

                if state['step'] > 0:
                    p.data = (1 - group['lr'] * group['weight_decay']) * p.data

                    state['M'] = beta1 * state['M'] + (1 - beta1) * grad
                    state['V'] = beta2 * state['V'] + (1 - beta2) * grad**2

                    M = state['M'] / (1 - beta1**state['step'])
                    V = state['V'] / (1 - beta2**state['step'])

                    p.data += -group['lr'] * M / (V**0.5 + group['eps'])

                state['step'] += 1
