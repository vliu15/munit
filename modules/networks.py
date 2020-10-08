# The MIT License
#
# Copyright (c) 2020 Vincent Liu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import torch
import torch.nn as nn

from modules.loss import GinormousCompositeLoss
from modules.layers import AdaptiveInstanceNorm2d, LayerNorm2d


class ResidualBlock(nn.Module):
    ''' Implements a residual block with (Adaptive) Instance Normalization '''

    def __init__(self, channels, s_dim=None, h_dim=None):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(
                nn.Conv2d(channels, channels, kernel_size=3)
            ),
        )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(
                nn.Conv2d(channels, channels, kernel_size=3)
            ),
        )
        self.use_style = s_dim is not None and h_dim is not None
        if self.use_style:
            self.norm1 = AdaptiveInstanceNorm2d(channels, s_dim, h_dim)
            self.norm2 = AdaptiveInstanceNorm2d(channels, s_dim, h_dim)
        else:
            self.norm1 = nn.InstanceNorm2d(channels)
            self.norm2 = nn.InstanceNorm2d(channels)

        self.activation = nn.ReLU()

    def forward(self, x, s=None):
        x_id = x
        x = self.conv1(x)
        x = self.norm1(x, s) if self.use_style else self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x, s) if self.use_style else self.norm2(x)
        return x + x_id


class ContentEncoder(nn.Module):
    ''' Implements a MUNIT encoder for content '''

    def __init__(self, base_channels=64, n_downsample=2, n_res_blocks=4):
        super().__init__()

        channels = base_channels

        # input convolutional layer
        layers = [
            nn.ReflectionPad2d(3),
            nn.utils.spectral_norm(
                nn.Conv2d(3, channels, kernel_size=7)
            ),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
        ]

        # downsampling layers
        for i in range(n_downsample):
            layers += [
                nn.ReflectionPad2d(1),
                nn.utils.spectral_norm(
                    nn.Conv2d(channels, 2 * channels, kernel_size=4, stride=2)
                ),
                nn.InstanceNorm2d(2 * channels),
                nn.ReLU(inplace=True),
            ]
            channels *= 2

        # residual blocks with non-adaptive instance normalization
        layers += [
            ResidualBlock(channels) for _ in range(n_res_blocks)
        ]
        self.layers = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x):
        return self.layers(x)

    @property
    def channels(self):
        return self.out_channels


class StyleEncoder(nn.Module):
    ''' Implements a MUNIT encoder for style '''

    n_deepen_layers = 2

    def __init__(self, base_channels=64, n_downsample=4, s_dim=8):
        super().__init__()

        channels = base_channels

        # input convolutional layer
        layers = [
            nn.ReflectionPad2d(3),
            nn.utils.spectral_norm(
                nn.Conv2d(3, channels, kernel_size=7, padding=0)
            ),
            nn.ReLU(inplace=True),
        ]

        # downsampling layers
        for i in range(self.n_deepen_layers):
            layers += [
                nn.ReflectionPad2d(1),
                nn.utils.spectral_norm(
                    nn.Conv2d(channels, 2 * channels, kernel_size=4, stride=2)
                ),
                nn.ReLU(inplace=True),
            ]
            channels *= 2
        for i in range(n_downsample - self.n_deepen_layers):
            layers += [
                nn.ReflectionPad2d(1),
                nn.utils.spectral_norm(
                    nn.Conv2d(channels, channels, kernel_size=4, stride=2)
                ),
                nn.ReLU(inplace=True),
            ]

        # apply global pooling and pointwise convolution to style_channels
        layers += [
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, s_dim, kernel_size=1),
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    '''
    Decoder Class
    Values:
        in_channels: number of channels from encoder output, a scalar
        n_upsample: number of upsampling layers, a scalar
        n_res_blocks: number of residual blocks, a scalar
        s_dim: the dimension of the style tensor (s), a scalar
        h_dim: the hidden dimension of the MLP, a scalar
    '''

    def __init__(self, in_channels, n_upsample=2, n_res_blocks=4, s_dim=8, h_dim=256):
        super().__init__()

        channels = in_channels

        # residual blocks with adaptive instance norm
        self.res_blocks = nn.ModuleList([
            ResidualBlock(channels, s_dim) for _ in range(n_res_blocks)
        ])

        # upsampling blocks
        layers = []
        for i in range(n_upsample):
            layers += [
                nn.Upsample(scale_factor=2),
                nn.ReflectionPad2d(2),
                nn.utils.spectral_norm(
                    nn.Conv2d(channels, channels // 2, kernel_size=5)
                ),
                LayerNorm2d(channels // 2),
            ]
            channels //= 2
        
        layers += [
            nn.ReflectionPad2d(3),
            nn.utils.spectral_norm(
                nn.Conv2d(channels, 3, kernel_size=7)
            ),
            nn.Tanh(),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x, s):
        for res_block in self.res_blocks:
            x = res_block(x, s=s)
        x = self.layers(x)
        return x


class Generator(nn.Module):
    ''' Implements a MUNIT generator '''

    def __init__(
        self,
        base_channels: int = 64,
        n_c_downsample: int = 2,
        n_s_downsample: int = 4,
        n_res_blocks: int = 4,
        s_dim: int = 8,
        h_dim: int = 256,
    ):
        super().__init__()
        self.c_enc = ContentEncoder(
            base_channels=base_channels, n_downsample=n_c_downsample, n_res_blocks=n_res_blocks,
        )
        self.s_enc = StyleEncoder(
            base_channels=base_channels, n_downsample=n_s_downsample, s_dim=s_dim,
        )
        self.dec = Decoder(
            self.c_enc.channels, n_upsample=n_c_downsample, n_res_blocks=n_res_blocks, s_dim=s_dim, h_dim=h_dim,
        )

    def encode(self, x):
        content = self.c_enc(x)
        style = self.s_enc(x)
        return (content, style)
    
    def decode(self, content, style):
        return self.dec(content, style)


class Discriminator(nn.Module):
    ''' Implements a MUNIT discriminator '''

    def __init__(
        self,
        base_channels: int = 64,
        n_layers: int = 3,
        n_discriminators: int = 3,
    ):
        super().__init__()

        self.discriminators = nn.ModuleList([
            self.patchgan_discriminator(base_channels, n_layers) for _ in range(n_discriminators)
        ])

        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    @staticmethod
    def patchgan_discriminator(base_channels, n_layers):
        '''
        Function that constructs and returns one PatchGAN discriminator module.
        '''
        channels = base_channels
        # input convolutional layer
        layers = [
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(
                nn.Conv2d(3, channels, kernel_size=4, stride=2),
            ),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # hidden convolutional layers
        for _ in range(n_layers):
            layers += [
                nn.ReflectionPad2d(1),
                nn.utils.spectral_norm(
                    nn.Conv2d(channels, 2 * channels, kernel_size=4, stride=2)
                ),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            channels *= 2

        # output projection layer
        layers += [
            nn.utils.spectral_norm(
                nn.Conv2d(channels, 1, kernel_size=1)
            ),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        outputs = []
        for discriminator in self.discriminators:
            outputs.append(discriminator(x))
            x = self.downsample(x)
        return outputs


class MUNIT(nn.Module):
    ''' Implements the MUNIT model in full '''

    def __init__(
        self,
        gen_channels: int = 64,
        n_c_downsample: int = 2,
        n_s_downsample: int = 4,
        n_res_blocks: int = 4,
        s_dim: int = 8,
        h_dim: int = 256,
        dis_channels: int = 64,
        n_layers: int = 3,
        n_discriminators: int = 3,
        scale_loss_weights_to_one: bool = True,
    ):
        super().__init__()

        self.gen_a = Generator(
            base_channels=gen_channels, n_c_downsample=n_c_downsample, n_s_downsample=n_s_downsample, n_res_blocks=n_res_blocks, s_dim=s_dim, h_dim=h_dim,
        )
        self.gen_b = Generator(
            base_channels=gen_channels, n_c_downsample=n_c_downsample, n_s_downsample=n_s_downsample, n_res_blocks=n_res_blocks, s_dim=s_dim, h_dim=h_dim,
        )
        self.dis_a = Discriminator(
            base_channels=dis_channels, n_layers=n_layers, n_discriminators=n_discriminators,
        )
        self.dis_b = Discriminator(
            base_channels=dis_channels, n_layers=n_layers, n_discriminators=n_discriminators,
        )
        self.s_dim = s_dim
        self.loss = GinormousCompositeLoss
        self.scale_loss_weights_to_one = scale_loss_weights_to_one

    def forward(self, x_a: torch.tensor, x_b: torch.tensor):
        s_a = torch.randn(x_a.size(0), self.s_dim, 1, 1, device=x_a.device).to(x_a.dtype)
        s_b = torch.randn(x_b.size(0), self.s_dim, 1, 1, device=x_b.device).to(x_b.dtype)

        # encode real x and compute image reconstruction loss
        x_a_loss, c_a, s_a_fake = self.loss.image_recon_loss(x_a, self.gen_a)
        x_b_loss, c_b, s_b_fake = self.loss.image_recon_loss(x_b, self.gen_b)

        # decode real (c, s) and compute latent reconstruction loss
        c_b_loss, s_a_loss, x_ba = self.loss.latent_recon_loss(c_b, s_a, self.gen_a)
        c_a_loss, s_b_loss, x_ab = self.loss.latent_recon_loss(c_a, s_b, self.gen_b)

        # compute adversarial losses
        gen_a_adv_loss = self.loss.adversarial_loss(x_ba, self.dis_a, False)
        gen_b_adv_loss = self.loss.adversarial_loss(x_ab, self.dis_b, False)

        # sum up losses for gen
        gen_loss = (
            10 * x_a_loss + c_b_loss + s_a_loss + gen_a_adv_loss + \
            10 * x_b_loss + c_a_loss + s_b_loss + gen_b_adv_loss
        )
        if self.scale_loss_weights_to_one:
            gen_loss = gen_loss * 0.1

        # sum up losses for dis
        dis_loss = (
            self.loss.adversarial_loss(x_ba.detach(), self.dis_a, False) + \
            self.loss.adversarial_loss(x_a.detach(), self.dis_a, True) + \
            self.loss.adversarial_loss(x_ab.detach(), self.dis_b, False) + \
            self.loss.adversarial_loss(x_b.detach(), self.dis_b, True)
        )

        return gen_loss, dis_loss, x_ab, x_ba

    def infer(self, x_a: torch.tensor, x_b: torch.tensor, encode_style: bool = True):
        self.eval()
        if not encode_style:
            s_a = torch.ones(x_a.shape, self.z_dim, 1, 1, device=x_a.device).to(x_a.dtype)
            s_b = torch.ones(x_b.shape, self.z_dim, 1, 1, device=x_b.device).to(x_b.dtype)

            c_a, _ = self.gen_a.encode(x_a)
            c_b, _ = self.gen_b.encode(x_b)
        else:
            c_a, s_a = self.gen_a.encode(x_a)
            c_b, s_b = self.gen_b.encode(x_b)

        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)

        return x_ba, x_ab
