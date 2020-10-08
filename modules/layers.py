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


class AdaptiveInstanceNorm2d(nn.Module):
    ''' Implements 2D Adaptive Instance Normalization '''

    def __init__(self, channels, s_dim=8, h_dim=256):
        super().__init__()

        self.instance_norm = nn.InstanceNorm2d(channels, affine=False)
        self.style_scale_transform = self.mlp(s_dim, h_dim, channels)
        self.style_shift_transform = self.mlp(s_dim, h_dim, channels)

    @staticmethod
    def mlp(self, in_dim, h_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, out_dim),
        )

    def forward(self, image, w):
        normalized_image = self.instance_norm(image)
        style_scale = self.style_scale_transform(w)[:, :, None, None]
        style_shift = self.style_shift_transform(w)[:, :, None, None]
        transformed_image = style_scale * normalized_image + style_shift
        return transformed_image


class LayerNorm2d(nn.Module):
    ''' Implements 2D Layer Normalization '''

    def __init__(self, channels, eps=1e-5, affine=True):
        super().__init__()
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.rand(channels))
            self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        mean = x.flatten(1).mean(1).reshape(-1, 1, 1, 1)
        std = x.flatten(1).std(1).reshape(-1, 1, 1, 1)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            x = x * self.gamma.reshape(1, -1, 1, 1) + self.beta.reshape(1, -1, 1, 1)

        return x
