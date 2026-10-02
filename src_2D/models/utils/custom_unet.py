from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_channels, affine=True)
        self.act1 = nn.LeakyReLU(0.1, inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.act2 = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Bilinear upsample completely eliminates checkerboard artifacts
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_up = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv_block = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.conv_up(x)
        
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            
        x = torch.cat([skip, x], dim=1)
        x = self.conv_block(x)
        return x

class CustomUNet(nn.Module):
    """
    A custom U-Net implementation utilizing Resize+Conv instead of ConvTranspose
    to mathematically eliminate checkerboard artifacts.
    Includes built-in dynamic padding for safe tensor sizing.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, filters: int = 16):
        super().__init__()
        if filters <= 0:
            raise ValueError("filters must be a positive integer")
            
        # 3 levels of 2x2 pooling means dimensions must be divisible by 8
        self._divisor = [8, 8]
        
        f = [filters, filters * 2, filters * 4, filters * 8]
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, f[0])
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = ConvBlock(f[0], f[1])
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = ConvBlock(f[1], f[2])
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(f[2], f[3])
        
        # Decoder
        self.dec3 = UpBlock(f[3], f[2])
        self.dec2 = UpBlock(f[2], f[1])
        self.dec1 = UpBlock(f[1], f[0])
        
        # Output
        self.out_conv = nn.Conv2d(f[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected input shape [B, C, H, W], got {x.shape}")
            
        original_shape = x.shape[-2:]
        x = self._pad_to_divisor(x)
        
        # Encoding
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool3(e3))
        
        # Decoding
        d3 = self.dec3(b, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        
        out = self.out_conv(d1)
        
        return self._crop_to_shape(out, original_shape)

    def _pad_to_divisor(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pad_h = (-h) % self._divisor[0]
        pad_w = (-w) % self._divisor[1]
        if pad_h or pad_w:
            # F.pad takes padding from the last dimension backwards: (W, H).
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x

    @staticmethod
    def _crop_to_shape(x: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        h, w = shape
        return x[..., :h, :w]
