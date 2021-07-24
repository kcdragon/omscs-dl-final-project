import torch
import torch.nn as nn

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