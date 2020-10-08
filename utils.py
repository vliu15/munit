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

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.utils import make_grid


def show_tensor_images(x_real, x_fake):
    ''' For visualizing images '''
    image_tensor = torch.cat((x_fake[:1, ...], x_real[:1, ...]), dim=0)
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat, nrow=1)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def weights_init(m):
    ''' Function for initializing weights with Kaiming normal '''
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')


def schedule(optimizers, decay=0.5):
    ''' Function for scheduling optimizer lr by decay factor '''
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay
