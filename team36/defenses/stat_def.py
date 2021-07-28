import torch
import torch.nn as nn
from . import detection_defense as dd


def out_size(image_size, convolution_layers):
    size = image_size
    for layer in convolution_layers:
        if isinstance(layer, nn.Conv2d):
            kernel_size = layer.kernel_size[0]
            stride = layer.stride[0]
            padding = layer.padding[0]
            size = (size + 2 * padding - kernel_size) // stride + 1
        elif isinstance(layer, nn.MaxPool2d):
            size = (size - layer.kernel_size) // layer.stride + 1
    return size


class VGG(nn.Module):
    def __init__(self, image_size=28, in_channels=1):
        super().__init__()

        num_classes = 10
        conv_kernel_size = 3
        conv_padding = 1
        conv_padding_mode = 'zeros'
        pool_kernel_size = 2
        pool_stride = 2

        self.convolution_layers = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=in_channels, out_channels=32,
                      kernel_size=conv_kernel_size, stride=1,
                      padding=conv_padding, padding_mode=conv_padding_mode),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=conv_kernel_size, stride=1,
                      padding=conv_padding, padding_mode=conv_padding_mode),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride),

            #Layer 2
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=conv_kernel_size, stride=1,
                      padding=conv_padding, padding_mode=conv_padding_mode),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride),
        )
        convolutional_layers_out_channels = 256
        convolutional_layers_out_size = out_size(image_size, self.convolution_layers)

        flattened_size = convolutional_layers_out_channels * pow(convolutional_layers_out_size, 2)
        self.linear_layers = nn.Sequential(
            nn.Linear(flattened_size, num_classes),
        )

    def forward(self, x):
        attack_array = torch.zeros((len(x)))
        for i in range(len(x)):
            attack = dd.detectAdversarial(x[i])
            attack_array[i] = attack
        valid_x = x[attack_array==0]
        out = self.convolution_layers(valid_x)
        out = out.reshape(out.shape[0], -1)
        out = self.linear_layers(out)
        return out
