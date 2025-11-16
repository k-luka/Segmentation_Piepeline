import torch.nn as nn
from model.UNet_Blocks import EncoderBlock, DecoderBlock, BottleNeck


class Unet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.enc1 = EncoderBlock(in_channels, 64, 3, 1, 0.1, 1)
        self.enc2 = EncoderBlock(64, 128, 3, 1, 0.1, 1)
        self.enc3 = EncoderBlock(128, 256, 3, 1, 0.1, 1)
        self.enc4 = EncoderBlock(256, 512, 3, 1, 0.1, 1)

        self.bottleneck = BottleNeck(512, 1024, 3, 1, 0.1, 1)

        self.dec1 = DecoderBlock(1024, 512, 3, 1, 0.1, 1)
        self.dec2 = DecoderBlock(512, 256, 3, 1, 0.1, 1)
        self.dec3 = DecoderBlock(256, 128, 3, 1, 0.1, 1)
        self.dec4 = DecoderBlock(128, 64, 3, 1, 0.1, 1)

        self.onebyone = nn.Conv2d(64, num_classes, 1, 1, 0)
    
    def forward(self, x):
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)
        
        x = self.bottleneck(x)

        x = self.dec1(skip4, x)
        x = self.dec2(skip3, x)
        x = self.dec3(skip2, x)
        x = self.dec4(skip1, x)

        out = self.onebyone(x)

        return out
    


