import copy
import torch.nn as nn

from .activation import MultiheadAttention, ReLU
from .dropout import Dropout
from .linear import Linear
from .normalization import LayerNorm

__all__ = [
    'TransformerDecoderLayer',
    'TransformerDecoder',
    'TransformerEncoderLayer',
    'TransformerEncoder'
]


class TransformerDecoderLayer(nn.Module):
    """TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network. 
    This standard decoder layer is based on the paper: Attention Is All You Need (https://arxiv.org/abs/1706.03762).
    
    Parameters
        d_model (int) - the number of expected features in the input (required).
        nhead (int) - the number of heads in the multiheadattention models (required).
        dim_feedforward (int) - the dimension of the feedforward network model (default=2048).
        dropout (float) - the dropout value (default=0.1).

    """

    # torch.nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=<function relu>, 
    #   layer_norm_eps=1e-05, batch_first=False, norm_first=False, bias=True, device=None, dtype=None)
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead)
        self.multihead_attn = MultiheadAttention(d_model, nhead)

        # feedforward network
        self.linear1 = Linear(d_model, dim_feedforward)
        self.activation = ReLU()
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

    # self-attention block
    def _sa_block(self, x, attn_mask, is_causal):
        x = self.self_attn(x, x, x, attn_mask, is_causal)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x, mem, attn_mask, is_causal):
        x = self.multihead_attn(x, mem, mem, attn_mask, is_causal)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

    # forward(tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, 
    #   memory_key_padding_mask=None, tgt_is_causal=False, memory_is_causal=False)
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_is_causal=False, memory_is_causal=False):
        """Pass the inputs (and mask) through the decoder layer.

        Parameters
            tgt (Tensor) - the sequence to the decoder layer (required).
            memory (Tensor) - the sequence from the last layer of the encoder (required).
            
        Shape
            (N, L, E)[tgt], (N, S, E)[memory] -> (N, L, E)

        """
        x = tgt
        x += self._sa_block(self.norm1(x), tgt_mask, tgt_is_causal)
        x += self._mha_block(self.norm2(x), memory, memory_mask, memory_is_causal)
        x += self._ff_block(self.norm3(x))

        return x


class TransformerDecoder(nn.Module):
    """TransformerDecoder is a stack of N decoder layers.

    Parameters
        decoder_layer - an instance of the TransformerDecoderLayer() class (required).
        num_layers - the number of sub-decoder-layers in the decoder (required).

    """
    # torch.nn.TransformerDecoder(decoder_layer, num_layers, norm=None)
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)

    # forward(tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, 
    #   memory_key_padding_mask=None, tgt_is_causal=None, memory_is_causal=False)
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_is_causal=None, memory_is_causal=False):
        """Pass the inputs (and mask) through the decoder layer in turn.

        Parameters
            tgt (Tensor) - the sequence to the decoder (required).
            memory (Tensor) - the sequence from the last layer of the encoder (required).

        Shape
            (N, L, E)[tgt], (N, S, E)[memory] -> (N, L, E)
        
        """
        for layer in self.layers:
            tgt = layer(
                tgt, 
                memory, 
                tgt_mask=tgt_mask, 
                memory_mask=memory_mask, 
                tgt_is_causal=tgt_is_causal, 
                memory_is_causal=memory_is_causal
            )

        return tgt


class TransformerEncoderLayer(nn.Module):
    """TransformerEncoderLayer is made up of self-attn and feedforward network. 
    This standard encoder layer is based on the paper: Attention Is All You Need (https://arxiv.org/abs/1706.03762).

    Parameters
        d_model (int) - the number of expected features in the input (required).
        nhead (int) - the number of heads in the multiheadattention models (required).
        dim_feedforward (int) - the dimension of the feedforward network model (default=2048).
        dropout (float) - the dropout value (default=0.1).

    """

    # torch.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=<function relu>, 
    #   layer_norm_eps=1e-05, batch_first=False, norm_first=False, bias=True, device=None, dtype=None)
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead)

        # feedforward network
        self.linear1 = Linear(d_model, dim_feedforward)
        self.activation = ReLU()
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    # self-attention block
    def _sa_block(self, x, attn_mask, is_causal):
        x = self.self_attn(x, x, x, attn_mask, is_causal)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, src, mask=None, is_causal=False):
        """Pass the input through the encoder layer.

        Parameters
            src (Tensor) - the sequence to the encoder layer (required).
            src_mask (Optional[Tensor]) - the mask for the src sequence (optional).
            is_causal (bool) - If specified, applies a causal mask as src mask. Default: False. 

        Shape
            (N, S, E) -> (N, S, E)

        """
        x = src
        x += self._sa_block(self.norm1(x), mask, is_causal)
        x += self._ff_block(self.norm2(x))

        return x


class TransformerEncoder(nn.Module):
    """TransformerEncoder is a stack of N encoder layers. 
    Users can build the BERT model with corresponding parameters (https://arxiv.org/abs/1810.04805).

    Parameters
        encoder_layer - an instance of the TransformerEncoderLayer() class (required).
        num_layers - the number of sub-encoder-layers in the encoder (required).

    Shape
        (N, S, E) -> (N, S, E)

    """

    # torch.nn.TransformerEncoder(encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True)
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)

    # forward(src, mask=None, src_key_padding_mask=None, is_causal=None)
    def forward(self, src, mask=None, is_causal=None):
        """Pass the input through the encoder layers in turn.

        Parameters
            src (Tensor) - the sequence to the encoder (required).
            mask (Optional[Tensor]) - the mask for the src sequence (optional).
            is_causal (Optional[bool]) - If specified, applies a causal mask as mask.
        """
        for layer in self.layers:
            src = layer(src, mask, is_causal)

        return src


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
