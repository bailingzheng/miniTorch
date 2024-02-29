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
        x, h_prev -> h

        x: (N, input_size) or (input_size)
        h_prev, h: (N, hidden_size) or (hidden_size)
    """

    # torch.nn.RNNCell(input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype=None)
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__(input_size, hidden_size, bias, num_chunks=1)

    def forward(self, x, h_prev=None):
        is_batched = x.dim() == 2

        if not is_batched:
            x = x.unsqueeze(0)

        if h_prev is None:
            h_prev = torch.zeros((x.size(0), self.hidden_size))
        else:
            h_prev = h_prev.unsqueeze(0) if not is_batched else h_prev
        # vf = _VF.rnn_tanh_cell(x, h_prev, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)

        v1 = x @ self.weight_ih.t()
        v2 = h_prev @ self.weight_hh.t()
        if self.bias:
            v1 += self.bias_ih
            v2 += self.bias_hh
        h = F.tanh(v1 + v2)

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
        x, h_prev -> h

        x: (N, input_size) or (input_size)
        h_prev, h: (N, hidden_size) or (hidden_size)
    """

    # torch.nn.GRUCell(input_size, hidden_size, bias=True, device=None, dtype=None)
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__(input_size, hidden_size, bias, num_chunks=3)

    def forward(self, x, h_prev=None):
        is_batched = x.dim() == 2

        if not is_batched:
            x = x.unsqueeze(0)

        if h_prev is None:
            h_prev = torch.zeros((x.size(0), self.hidden_size))
        else:
            h_prev = h_prev.unsqueeze(0) if not is_batched else h_prev
        # vf = _VF.gru_cell(x, h_prev, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)

        v1 = x @ self.weight_ih.t()
        v2 = h_prev @ self.weight_hh.t()
        if self.bias:
            v1 += self.bias_ih
            v2 += self.bias_hh
        l1 = v1.chunk(3, dim=1)
        l2 = v2.chunk(3, dim=1)

        r = F.sigmoid(l1[0] + l2[0])
        z = F.sigmoid(l1[1] + l2[1])
        n = F.tanh(l1[2] + r*l2[2])
        h = (1 - z)*n + z*h_prev

        if not is_batched:
            h = h.squeeze(0)
        # print((vf - h).abs().max())
        return h


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
        x, (h_prev, c_prev) -> (h, c)

        x: (N, input_size) or (input_size)
        h_prev, c_prev, h, c: (N, hidden_size) or (hidden_size)
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__(input_size, hidden_size, bias, num_chunks=4)

    def forward(self, x, v_prev=None):
        is_batched = x.dim() == 2

        if not is_batched:
            x = x.unsqueeze(0)

        if v_prev is None:
            h_prev = torch.zeros((x.size(0), self.hidden_size))
            c_prev = torch.zeros((x.size(0), self.hidden_size))
        else:
            h_prev = v_prev[0].unsqueeze(0) if not is_batched else v_prev[0]
            c_prev = v_prev[1].unsqueeze(0) if not is_batched else v_prev[1]
        # vf = _VF.lstm_cell(x, (h_prev, c_prev), self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)

        v1 = x @ self.weight_ih.t()
        v2 = h_prev @ self.weight_hh.t()
        if self.bias:
            v1 += self.bias_ih
            v2 += self.bias_hh
        l1 = v1.chunk(4, dim=1)
        l2 = v2.chunk(4, dim=1)

        i = F.sigmoid(l1[0] + l2[0])
        f = F.sigmoid(l1[1] + l2[1])
        g = F.tanh(l1[2] + l2[2])
        o = F.sigmoid(l1[3] + l2[3])
        c = f*c_prev + i*g
        h = o * F.tanh(c)

        if not is_batched:
            h = h.squeeze(0)
            c = c.squeeze(0)
        # print((vf[0] - h).abs().max())
        # print((vf[1] - c).abs().max())
        return (h, c)