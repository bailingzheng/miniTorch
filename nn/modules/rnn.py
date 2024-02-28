import math
import torch
import torch._VF as _VF
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'GRUCell',
    'LSTMCell',
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
        x, h0 -> h1

        x: (N, input_size) or (input_size)
        h0: (N, hidden_size) or (hidden_size)
        h1: (N, hidden_size) or (hidden_size)
    """

    # torch.nn.RNNCell(input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype=None)
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__(input_size, hidden_size, bias, num_chunks=1)

    def forward(self, x, h=None):
        is_batched = x.dim() == 2

        if not is_batched:
            x = x.unsqueeze(0)

        if h is None:
            h0 = torch.zeros((x.size(0), self.hidden_size))
        else:
            h0 = h.unsqueeze(0) if not is_batched else h
        # vf = _VF.rnn_tanh_cell(x, h0, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)

        p1 = x @ self.weight_ih.t()
        p2 = h0 @ self.weight_hh.t()
        if self.bias:
            p1 += self.bias_ih
            p2 += self.bias_hh
        h1 = F.tanh(p1 + p2)

        if not is_batched:
            h1 = h1.squeeze(0)
        # print((vf - h1).abs().max())
        return h1

        
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
        x, h0 -> h1

        x: (N, input_size) or (input_size)
        h0: (N, hidden_size) or (hidden_size)
        h1: (N, hidden_size) or (hidden_size)
    """

    # torch.nn.GRUCell(input_size, hidden_size, bias=True, device=None, dtype=None)
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__(input_size, hidden_size, bias, num_chunks=3)

    def forward(self, x, h=None):
        is_batched = x.dim() == 2

        if not is_batched:
            x = x.unsqueeze(0)

        if h is None:
            h0 = torch.zeros((x.size(0), self.hidden_size))
        else:
            h0 = h.unsqueeze(0) if not is_batched else h
        # vf = _VF.gru_cell(x, h0, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)

        p1 = x @ self.weight_ih.t()
        p2 = h0 @ self.weight_hh.t()
        if self.bias:
            p1 += self.bias_ih
            p2 += self.bias_hh

        r = F.sigmoid(p1.chunk(3, dim=1)[0] + p2.chunk(3, dim=1)[0])
        z = F.sigmoid(p1.chunk(3, dim=1)[1] + p2.chunk(3, dim=1)[1])
        n = F.tanh(p1.chunk(3, dim=1)[2] + r*(p2.chunk(3, dim=1)[2]))
        h1 = (1 - z)*n + z*h0

        if not is_batched:
            h1 = h1.squeeze(0)
        # print((vf - h1).abs().max())
        return h1


class LSTMCell(RNNCellBase):
    """A long short-term memory (LSTM) cell.

    i = sigmoid(x@W_ii.T + b_ii + h@W_hi.T + b_hi)
    f = sigmoid(x@W_if.T + b_if + h@W_hf.T + b_hf)
    g = tanh(x@W_ig.T + b_ig + h@W_hg.T + b_hg)
    o = sigmoid(x@W_io.T + b_io + h@W_ho.T + b_ho)
    c = f*c + i*g
    h = o * tanh(c)

    Parameters
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights. Default: True

    Shape
        x, (h0, c0) -> (h1, c1)

        x: (N, input_size) or (input_size)
        h0: (N, hidden_size) or (hidden_size)
        c0: (N, hidden_size) or (hidden_size)
        h1: (N, hidden_size) or (hidden_size)
        c1: (N, hidden_size) or (hidden_size)
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__(input_size, hidden_size, bias, num_chunks=4)

    def forward(self, x, h=None):
        is_batched = x.dim() == 2

        if not is_batched:
            x = x.unsqueeze(0)

        if h is None:
            h0 = torch.zeros((x.size(0), self.hidden_size))
            c0 = torch.zeros((x.size(0), self.hidden_size))
        else:
            (h0, c0) = (h[0].unsqueeze(0), h[1].unsqueeze(0)) if not is_batched else h
        # vf = _VF.lstm_cell(x, (h0, c0), self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)

        p1 = x @ self.weight_ih.t()
        p2 = h0 @ self.weight_hh.t()
        if self.bias:
            p1 += self.bias_ih
            p2 += self.bias_hh

        i = F.sigmoid(p1.chunk(4, dim=1)[0] + p2.chunk(4, dim=1)[0])
        f = F.sigmoid(p1.chunk(4, dim=1)[1] + p2.chunk(4, dim=1)[1])
        g = F.tanh(p1.chunk(4, dim=1)[2] + p2.chunk(4, dim=1)[2])
        o = F.sigmoid(p1.chunk(4, dim=1)[3] + p2.chunk(4, dim=1)[3])
        c1 = f*c0 + i*g
        h1 = o * F.tanh(c1)

        if not is_batched:
            (h1, c1) = (h1.squeeze(0), c1.squeeze(0))
        # print((vf[0] - h1).abs().max())
        # print((vf[1] - c1).abs().max())
        return (h1, c1)