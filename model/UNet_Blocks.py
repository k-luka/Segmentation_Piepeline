import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropoutprob, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout2d(dropoutprob)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        preactivation = x
        postactivation = self.activation(x)
        x = self.dropout(postactivation)
        return preactivation, postactivation

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropoutprob, padding):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size, stride, dropoutprob, padding)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size, stride, dropoutprob, padding)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        _, x = self.conv1(x)
        skip, x = self.conv2(x)
        x = self.pool(x)
        return skip, x
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropoutprob, padding):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size, stride, dropoutprob, padding)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size, stride, dropoutprob, padding)
    
    def forward(self, skip, x):
        x = self.upconv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = torch.cat((x, skip), dim=1)
        _, x = self.conv1(x)
        _, x = self.conv2(x)

        return x
    

class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropoutprob, padding):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size, stride, dropoutprob, padding)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size, stride, dropoutprob, padding)
    
    def forward(self, x):
        _, x = self.conv1(x)
        _, x = self.conv2(x)
        return x
    







