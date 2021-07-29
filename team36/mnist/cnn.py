import torch
import torch.nn as nn
from .cnn_util import out_size

class CNN(nn.Module):
    """
    Simple CNN 
    1 conv layer -> relu -> maxpool -> 1 fully connected layer
    Copied from vgg.py with extra layers removed
    """
    def __init__(self, image_size=28, in_channels=1):
        super().__init__()

        image_size = image_size
        num_classes = 10
        conv_kernel_size = 3
        conv_padding = 0
        conv_padding_mode = 'zeros'
        pool_kernel_size = 2
        pool_stride = 2

        self.convolution_layers = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=in_channels, out_channels=32,
                      kernel_size=conv_kernel_size, stride=1,
                      padding=conv_padding, padding_mode=conv_padding_mode),
#             nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=32, out_channels=64,
#                       kernel_size=conv_kernel_size, stride=1,
#                       padding=conv_padding, padding_mode=conv_padding_mode),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride),

            # Layer 2
#             nn.Conv2d(in_channels=64, out_channels=128,
#                       kernel_size=conv_kernel_size, stride=1,
#                       padding=conv_padding, padding_mode=conv_padding_mode),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=128, out_channels=256,
#                       kernel_size=conv_kernel_size, stride=1,
#                       padding=conv_padding, padding_mode=conv_padding_mode),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride),
        )
        convolutional_layers_out_channels = 32
        convolutional_layers_out_size = out_size(image_size, self.convolution_layers)

        flattened_size = convolutional_layers_out_channels * pow(convolutional_layers_out_size, 2)
        self.linear_layers = nn.Sequential(
            nn.Linear(flattened_size, num_classes),
        )

    def forward(self, x):
        out = self.convolution_layers(x)
        out = out.reshape(out.shape[0], -1)
        out = self.linear_layers(out)
        return out