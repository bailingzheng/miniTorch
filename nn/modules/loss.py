import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'CrossEntropyLoss',
    'L1Loss',
    'MSELoss',
    'NLLLoss',
    'TripletMarginLoss'
]

class _Loss(nn.Module):

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction


class _WeightedLoss(_Loss):

    def __init__(self, weight=None, reduction='mean'):
        super().__init__(reduction)
        self.weight = weight


class L1Loss(_Loss):
    """Creates a criterion that measures the mean absolute error (MAE) between each element in
    the input x and target y.

    L = {l[1], l[2], ..., l[N]}, l[n] = |x[n] - y[n]|

    Parameters
        reduction - Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 

    Shape:
        Input: (*), where * means any number of dimensions.
        Target: (*), same shape as the input.
        Output: scalar. If reduction is 'none', then (*), same shape as the input.
    """

    def __init__(self, reduction='mean'):
        super().__init__(reduction)

    def forward(self, input, target):
        L = (input - target).abs()

        if self.reduction == 'mean':
            output = L.mean()
        elif self.reduction == 'sum':
            output = L.sum()
        else:
            output = L
        # f = F.l1_loss(input, target, reduction=self.reduction)
        # print((f - output).abs().max())
        return output


class MSELoss(_Loss):
    """Creates a criterion that measures the mean squared error (squared L2 norm) between each element in 
    the input x and target y.

    L = {l[1], l[2], ..., l[N]}, l[n] = (x[n] - y[n])**2

    Parameters
        reduction - Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 

    Shape:
        Input: (*), where * means any number of dimensions.
        Target: (*), same shape as the input.
        Output: scalar. If reduction is 'none', then (*), same shape as the input.
    """

    def __init__(self, reduction='mean'):
        super().__init__(reduction)

    def forward(self, input, target):
        L = (input - target).square()

        if self.reduction == 'mean':
            output = L.mean()
        elif self.reduction == 'sum':
            output = L.sum()
        else:
            output = L
        # f = F.mse_loss(input, target, reduction=self.reduction)
        # print((f - output).abs().max())
        return output


class CrossEntropyLoss(_WeightedLoss):
    """This criterion computes the cross entropy loss between input logits and target.

    The input is expected to contain the unnormalized logits for each class (which do not need to be positive or sum to 1, in general).

    The target that this criterion expects should contain probabilities for each class.

    L = {l[1], l[2], ..., l[N]}, l[n] = -sum(y[n] * log_softmax(x[n]))

    Parameters
        reduction - Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 

    Shape
        Input: (C) or (N, C).
        Target: If containing class probabilities, same shape as the input and each value should be between [0, 1].
        Output: If reduction is 'none', shape (), (N), depending on the shape of the input. Otherwise, scalar.

        where C is the number of classes, and N is the batch size.
    """

    def __init__(self, reduction='mean'):
        super().__init__(weight=None, reduction=reduction)

    def forward(self, input, target):
        dim = 0 if input.dim() == 1 else 1
        L = -(target * input.log_softmax(dim=dim)).sum(dim=dim)

        if self.reduction == 'mean':
            output = L.mean()
        elif self.reduction == 'sum':
            output = L.sum()
        else:
            output = L
        # f = F.cross_entropy(input, target, reduction=self.reduction)
        # print((f - output).abs().max())
        return output


class NLLLoss(_WeightedLoss):
    """The negative log likelihood loss. It is useful to train a classification problem with C classes.

    The input given through a forward call is expected to contain log-probabilities of each class.

    Obtaining log-probabilities in a neural network is easily achieved by adding a LogSoftmax layer in the last layer of your network. 
    You may use CrossEntropyLoss instead, if you prefer not to add an extra layer.

    The target that this loss expects should be a class index in the range [0, C-1].

    L = {l[1], l[2], ..., l[N]}, l[n] = -x[n, y[n]] 

    Parameters
        reduction - Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 

    Shape
        Input: (N, C) or (C).
        Target: (N) or ().
        Output: If reduction is 'none', shape (N). Otherwise, scalar.

        where C is the number of classes, and N is the batch size.
    """

    def __init__(self, reduction='mean'):
        super().__init__(weight=None, reduction=reduction)

    def forward(self, input, target):
        if input.dim() == 1:
            L = -input[target]  
        else:
            N, _ = input.shape
            L = -input[range(N), target]

        if self.reduction == 'mean':
            output = L.mean()
        elif self.reduction == 'sum':
            output = L.sum()
        else:
            output = L
        # f = F.nll_loss(input, target, reduction=self.reduction)
        # print((f - output).abs().max())
        return output


class TripletMarginLoss(_Loss):
    """Creates a criterion that measures the triplet loss given an input tensors x1, x2, x3, and a margin with a value greater than 0. 
    This is used for measuring a relative similarity between samples. 

    L(a, p, n) = max{d(a, p) - d(a, n) + margin, 0}

    Parameters
        margin - 
        p - The norm degree for pairwise distance.
        eps - Small constant for numerical stability. 
        reduction - Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 

    Shape
        Input: (N, D) or (D) where D is the vector dimension.
        Output: If reduction is 'none' and input shape is (N, D), shape (N). Otherwise, scalar.
    """

    # torch.nn.TripletMarginLoss(margin=1.0, p=2.0, eps=1e-06, swap=False, size_average=None, reduce=None, reduction='mean')
    def __init__(self, margin=1.0, p=2.0, eps=1e-06, reduction='mean'):
        super().__init__(reduction)
        self.margin = margin
        self.p = p
        self.eps = eps

    def forward(self, anchor, positive, negative):
        dim = 0 if anchor.dim() == 1 else 1

        d1 = torch.linalg.vector_norm(anchor - positive + self.eps, ord=self.p, dim=dim)
        d2 = torch.linalg.vector_norm(anchor - negative + self.eps, ord=self.p, dim=dim)
        L = d1 - d2 + self.margin
        L = torch.maximum(L, torch.zeros_like(L))
        
        if self.reduction == 'mean':
            output = L.mean()
        elif self.reduction == 'sum':
            output = L.sum()
        else:
            output = L
        # f = F.triplet_margin_loss(anchor, positive, negative, margin=self.margin, p=self.p, eps=self.eps, reduction=self.reduction)
        # print((f - output).abs().max())
        return output