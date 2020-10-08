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
import torch.nn.functional as F


class GinormousCompositeLoss(nn.Module):
    ''' Implements all base losses for MUNIT '''

    @staticmethod
    def image_recon_loss(x, gen):
        c, s = gen.encode(x)
        recon = gen.decode(c, s)
        return F.l1_loss(recon, x), c, s

    @staticmethod
    def latent_recon_loss(c, s, gen):
        x_fake = gen.decode(c, s)
        recon = gen.encode(x_fake)
        return F.l1_loss(recon[0], c), F.l1_loss(recon[1], s), x_fake

    @staticmethod
    def adversarial_loss(x, dis, is_real):
        preds = dis(x)
        target = torch.ones_like if is_real else torch.zeros_like
        loss = 0.0
        for pred in preds:
            loss += F.mse_loss(pred, target(pred))
        return loss
