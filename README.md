# miniTorch

Inspired by [Andrej Karpathy's course](https://karpathy.ai/zero-to-hero.html).  
I'm trying to implement a much simplified version of torch.nn, and torch.optim.  
The APIs are almost the same as PyTorch, except that some optional parameters are not supported.

## Neural Network
- Convolution Layers
    * Conv2d
- Dropout Layers
    * Dropout
- Linear Layers
    * Linear
- Normalization Layers
    * BatchNorm1d
    * BatchNorm2d
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

## Optimizers
- Adam
- AdamW
- RMSprop
- SGD

## Computer Vision Models
- Classification
    * ResNet
    * MobileNetV2
- Object Detection
    * YOLOv1
- Semantic Segmentation
    * UNet

## Language Models
- Bigram
- MLP
- Transformer
