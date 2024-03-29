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
- Loss Functions
    * CrossEntropyLoss
    * L1Loss
    * MSELoss
    * NLLLoss
    * TripletMarginLoss
- Normalization Layers
    * BatchNorm1d
    * BatchNorm2d
    * LayerNorm
- Non-linear Activations
    * Hardtanh
    * LeakyReLU
    * LogSoftmax
    * MultiheadAttention
    * ReLU
    * ReLU6
    * Tanh
- Pooling layers
    * MaxPool2d
- Recurrent Layers
    * GRUCell
    * LSTMCell
    * RNNCell
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
- SGD (with momentum)

## Text Transforms
- GPT2BPETokenizer

## Convolutional Neural Networks
- AlexNet
- GoogLeNet
- InceptionResnetV1
- MobileNetV2
- ResNet
- VGG
- YOLOv1
- UNet
- DCGAN

## Sequence Models
- Bigram
- MLP
- RNN
- Transformer