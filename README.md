# miniTorch

Inspired by [Andrej Karpathy's course](https://karpathy.ai/zero-to-hero.html).  
I'm trying to implement a much simplified version of torch.nn, torch.optim, and torchvision.models.  
The APIs are almost the same as PyTorch, except that some optional parameters are not supported.

[torch.nn](https://pytorch.org/docs/stable/nn.html): 
- Convolution Layers
    * Conv2d
- Dropout Layers
    * Dropout
- Linear Layers
    * Linear
- Normalization Layers
    * BatchNorm1d
    * LayerNorm
- Non-linear Activations
    * MultiheadAttention
    * ReLU
    * Tanh
- Pooling layers
    * MaxPool2d
- Sparse Layers
    * Embedding
- Transformer Layers
    * TransformerDecoderLayer
    * TransformerDecoder
    * TransformerEncoderLayer
    * TransformerEncoder
    * Transformer
- Utilities
    * Flatten

[torch.optim](https://pytorch.org/docs/stable/optim.html):
- Adam
- AdamW
- RMSprop
- SGD

[torchvision.models](https://pytorch.org/vision/stable/models.html)
- Classification
    * ResNet
    * MobileNetV2