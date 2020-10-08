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

import argparse

from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch
import yaml

from utils import show_tensor_images


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.yml')
    parser.add_argument('-s', '--encode_style', action='store_true', default=False)
    parser.add_argument('-n', '--show_n', type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_arguments()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        config = OmegaConf.create(config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    munit = instantiate(config.munit).to(device)
    munit.load_state_dict(torch.load(config.resume_checkpoint)['munit_model_dict'])
    dataloader = instantiate(config.test_dataloader, instantiate(config.test_dataset))

    n = 0
    for (x_a, x_b) in dataloader:
        if n == args.show_n:
            break

        x_a = x_a.to(device)
        x_b = x_b.to(device)

        x_ba, x_ab = munit.infer(x_a, x_b, encode_style=args.encode_style)
        show_tensor_images(x_a, x_ab.to(x_a.dtype))
        show_tensor_images(x_b, x_ba.to(x_b.dtype))

        n += 1


if __name__ == '__main__':
    main()
