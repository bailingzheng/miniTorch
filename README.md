# miniTorch

Inspired by [Andrej Karpathy's course](https://karpathy.ai/zero-to-hero.html).  
I'm trying to implement a much simplified version of torch.nn and torch.optim.  
The APIs are almost the same as PyTorch, except that some optional parameters are not supported.

[torch.nn](https://pytorch.org/docs/stable/nn.html): 
- Activation Layers (MultiheadAttention, ReLU, Tanh)
- Convolution Layers (Conv2d)
- Dropout Layers (Dropout)
- Linear Layers (Linear)
- Normalization Layers (BatchNorm1d, LayerNorm)
- Pooling layers (MaxPool2d)
- Sparse Layers (Embedding)
- Utilities (Flatten)


[torch.optim](https://pytorch.org/docs/stable/optim.html)
- Adam
- RMSprop
- SGD