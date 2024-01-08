# miniTorch

Inspired by [Andrej Karpathy's course](https://karpathy.ai/zero-to-hero.html).  
I'm trying to implement a much simplified version of [torch.nn](https://pytorch.org/docs/stable/nn.html).  
The APIs are almost the same as PyTorch, except that some optional parameters are not supported.

APIs that are implemented so far: 
- Activation Layers (MultiHeadAttention, ReLU, Tanh)
- Convolution Layers (Conv2d)
- Dropout Layers (Dropout)
- Linear Layers (Linear)
- Normalization Layers (BatchNorm1d, LayerNorm)
- Sparse Layers (Embedding)
- Utilities (Flatten)