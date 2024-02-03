import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from math import floor


class Generator(torch.nn.Module):

    def __init__(self, kernel_size, attention_layer, attention_positions):
        super().__init__()
        self.EncodeLayers = torch.nn.ModuleList()
        self.DecodeLayers = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        
        self.EncoderChannels = [1, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
        self.DecoderChannels = [2048, 1024, 512, 512, 256, 256, 128, 128, 64, 64, 32, 1]
        
        self.attention_layer = attention_layer
        self.enc_attn_positions = attention_positions
        self.dec_attn_positions = [6-x for x in attention_positions]
        
        # Create encoder and decoder layers.
        for i in range(len(self.EncoderChannels) - 1):
            outs = self.EncoderChannels[i + 1]
            self.EncodeLayers.append(
                nn.Conv1d(
                    in_channels=self.EncoderChannels[i],
                    out_channels=outs,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=floor(kernel_size / 2),  # same
                )
            )

        for i in range(len(self.DecoderChannels) - 1):
            ins = self.EncoderChannels[-1 * (i + 1)] * 2
            self.DecodeLayers.append(
                nn.ConvTranspose1d(
                    in_channels=ins,
                    out_channels=self.EncoderChannels[-1 * (i + 2)],
                    kernel_size=kernel_size + 1,  # adding one to kernel size makes the dimensions match
                    stride=2,
                    padding=floor(kernel_size / 2),  # same
                )
            )

    def forward(self, x):
        """Forward pass through autoencoder"""
        # encode
        skips = []
        x = x.permute(0, 2, 1)
        for i, layer in enumerate(self.EncodeLayers):
#             print(f"Generator: Encoder: Layer: {i} - Input Shape: {x.shape} - C: {self.EncoderChannels[i+1]}")
            x = layer(x)
            skips.append(x.clone())
            if i == len(self.DecodeLayers) - 1:
                
                # Attention Layer
                if i in self.enc_attn_positions:
#                     print(f"==> Encoder Attention layer:{i} Input Shape: {x.shape}, C: {self.EncoderChannels[i+1]}")
                    attn_layer = self.attention_layer(C=self.EncoderChannels[i+1])
                    x = attn_layer(x)
                continue
            else:
                x = F.leaky_relu(x, negative_slope=0.3)
                
            # Attention Layer
            if i in self.enc_attn_positions:
#                 print(f"===> Encoder Attention layer:{i} Input Shape: {x.shape}, C: {self.EncoderChannels[i+1]}")
                attn_layer = self.attention_layer(C=self.EncoderChannels[i+1])
                x = attn_layer(x)

        # fuse z
        z = torch.normal(torch.zeros_like(x), torch.ones_like(x))
        x = torch.cat((x, z), 1)

        # decode
        for i, layer in enumerate(self.DecodeLayers):
#             print(f"Generator: Decoder: Layer: {i} ==")
            x = layer(x)
            if i == len(self.DecodeLayers) - 1:
                # Attention Layer
                if i in self.dec_attn_positions:
#                     print(f"==> Decoder Attention: layer {i} Input Shape: {x.shape}, C: {self.EncoderChannels[-1 * (i + 2)]*2}")
                    attn_layer = self.attention_layer(C=self.EncoderChannels[-1 * (i + 2)]*2)
                    x = attn_layer(x)
                continue
            else:
                x = torch.cat((x, skips[-1 * (i + 2)]), 1)
                x = F.leaky_relu(x, negative_slope=0.3)
                
            # Attention Layer
            if i in self.dec_attn_positions:
#                 print(f"===> Decoder Attention: layer {i} Input Shape: {x.shape}, C: {self.EncoderChannels[-1 * (i + 2)]*2}")
                attn_layer = self.attention_layer(C=self.EncoderChannels[-1 * (i + 2)]*2)
                x = attn_layer(x)
                    
        x = x.permute(0, 2, 1)
        return x
    

class Discriminator(torch.nn.Module):

    def __init__(self, kernel_size, attention_layer, attention_positions):
        super().__init__()
        self.Layers = torch.nn.ModuleList()
        self.Norms = torch.nn.ModuleList()
        
        self.attention_layer = attention_layer
        self.attention_positions = attention_positions
        
        self.Channels = [2, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024, 1]
        
        # Create encoder and decoder layers.
        for i in range(len(self.Channels) - 1):
            if i != len(self.Channels) - 2:
                self.Layers.append(
                    nn.Conv1d(
                        in_channels=self.Channels[i],
                        out_channels=self.Channels[i + 1],
                        kernel_size=kernel_size,
                        stride=2,
                        padding=floor(kernel_size / 2),  # same
                    )
                )
                self.Norms.append(
                    nn.BatchNorm1d(
                        num_features=self.Channels[
                            i + 1
                        ]
                    )
                )
            # output convolution
            else:
                self.Layers.append(
                    nn.Conv1d(
                        in_channels=self.Channels[i],
                        out_channels=self.Channels[i + 1],
                        kernel_size=1,
                        stride=1,
                        padding=0,  # same
                    )
                )
                self.Layers.append(
                    nn.Linear(in_features=8, out_features=1,)  # Channels[i+1],
                )

    def forward(self, x):
        """forward pass through the discriminator"""
        x = x.permute(0, 2, 1)
        # encode
        for i in range(len(self.Norms)):
#             print(f"Discriminator: Decoder: {i}")
            x = self.Layers[i](x)
            x = self.Norms[i](x)
            x = F.leaky_relu(x, negative_slope=0.3)
            
            # Attention Layer
            if i in self.attention_positions:
#                 print(f"Discriminator: Decoder: Attention: {i}, inp_shape: {x.shape}")
                attn_layer = self.attention_layer(C=self.Channels[i + 1])
                x = attn_layer(x)

        # output
        x = self.Layers[-2](x)
        x = self.Layers[-1](x)
        # x = F.sigmoid(x)
        x = x.permute(0, 2, 1)

        return x  # in logit format