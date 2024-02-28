import math
import torch
import torch._VF as _VF
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'GRUCell',
    'RNNCell'
]


class RNNCellBase(nn.Module):

    def __init__(self, input_size, hidden_size, bias, num_chunks):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.weight_ih = nn.Parameter(torch.empty((num_chunks * hidden_size, input_size)))
        self.weight_hh = nn.Parameter(torch.empty((num_chunks * hidden_size, hidden_size)))

        if bias:
            self.bias_ih = nn.Parameter(torch.empty((num_chunks * hidden_size)))
            self.bias_hh = nn.Parameter(torch.empty((num_chunks * hidden_size)))
        else:
            self.bias_ih = None
            self.bias_hh = None

        k = 1.0 / hidden_size if hidden_size > 0 else 0
        with torch.no_grad():
            for p in self.parameters():
                p.uniform_(-math.sqrt(k), math.sqrt(k))


class RNNCell(RNNCellBase):
    """An Elman RNN cell with tanh non-linearity.

    h = tanh(x@W_ih.T + b_ih + h@W_hh.T + b_hh)

    Parameters
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights b_ih and b_hh. Default: True

    Shape
        x: (N, input_size) or (input_size)
        h: (N, hidden_size) or (hidden_size)
    """

    # torch.nn.RNNCell(input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype=None)
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__(input_size, hidden_size, bias, num_chunks=1)

    def forward(self, x, h=None):
        is_batched = x.dim() == 2

        if not is_batched:
            x = x.unsqueeze(0)

        if h is None:
            h = torch.zeros((x.size(0), self.hidden_size))
        else:
            h = h.unsqueeze(0) if not is_batched else h
        # vf = _VF.rnn_tanh_cell(x, h, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)

        p1 = x @ self.weight_ih.t()
        p2 = h @ self.weight_hh.t()
        if self.bias:
            p1 += self.bias_ih
            p2 += self.bias_hh
        h = F.tanh(p1 + p2)

        if not is_batched:
            h = h.squeeze(0)
        # print((vf - h).abs().max())
        return h

        
class GRUCell(RNNCellBase):
    """A gated recurrent unit (GRU) cell.

    r = sigmoid(x@W_ir.T + b_ir + h@W_hr.T + b_hr)
    z = sigmoid(x@W_iz.T + b_iz + h@W_hz.T + b_hz)
    n = tanh(x@W_in.T + b_in + r*(h@W_hn.T + b_hn))
    h = (1 - z)*n + z*h

    Parameters
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights. Default: True

    Shape
        x: (N, input_size) or (input_size)
        h: (N, hidden_size) or (hidden_size)
    """

    # torch.nn.GRUCell(input_size, hidden_size, bias=True, device=None, dtype=None)
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__(input_size, hidden_size, bias, num_chunks=3)

    def forward(self, x, h=None):
        is_batched = x.dim() == 2

        if not is_batched:
            x = x.unsqueeze(0)

        if h is None:
            h = torch.zeros((x.size(0), self.hidden_size))
        else:
            h = h.unsqueeze(0) if not is_batched else h
        # vf = _VF.gru_cell(x, h, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)

        p1 = x @ self.weight_ih.t()
        p2 = h @ self.weight_hh.t()
        if self.bias:
            p1 += self.bias_ih
            p2 += self.bias_hh

        r = F.sigmoid(p1.chunk(3, dim=1)[0] + p2.chunk(3, dim=1)[0])
        z = F.sigmoid(p1.chunk(3, dim=1)[1] + p2.chunk(3, dim=1)[1])
        n = F.tanh(p1.chunk(3, dim=1)[2] + r*(p2.chunk(3, dim=1)[2]))
        h = (1 - z)*n + z*h

        if not is_batched:
            h = h.squeeze(0)
        # print((vf - h).abs().max())
        return h
