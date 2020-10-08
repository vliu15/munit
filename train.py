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
from datetime import datetime
import os

from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
import yaml

from utils import weights_init, schedule


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.yml')
    return parser.parse_args()


def train(munit, dataloader, optimizers, train_config, device, start_step):
    gen_optimizer, dis_optimizer = optimizers

    # initialize logging
    log_dir = os.path.join(train_config.log_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(log_dir, mode=0o775, exist_ok=False)

    cur_step = 0

    while cur_step < train_config.max_steps:

        gen_losses = 0.0
        dis_losses = 0.0
        epoch_steps = 0
        pbar = tqdm(dataloader, position=0, desc='train [G loss: -.----][D loss: -.----]')
        for (x_a, x_b) in pbar:
            x_a = x_a.to(device)
            x_b = x_b.to(device)

            with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                outputs = munit(x_a, x_b)
            
            gen_loss, dis_loss, x_ab, x_ba = outputs
            munit.zero_grad()

            gen_optimizer.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()

            dis_optimizer.zero_grad()
            dis_loss.backward()
            dis_optimizer.step()

            gen_losses += gen_loss.item()
            dis_losses += dis_loss.item()

            # update progress
            cur_step += 1
            epoch_steps += 1
            pbar.set_description(
                desc=f'train [G loss: {gen_losses/epoch_steps:.4f}][D loss: {dis_losses/epoch_steps:.4f}]'
            )

            # save if necessary
            if cur_step % train_config.save_every == 0:
                torch.save({
                    'munit_model_dict': munit.state_dict(),
                    'g_optim_dict': gen_optimizer.state_dict(),
                    'd_optim_dict': dis_optimizer.state_dict(),
                    'step': cur_step,
                }, os.path.join(log_dir, f'step={cur_step}.pt'))

            # schedule learning rate if necessary
            if cur_step % train_config.decay_every == 0:
                schedule([gen_optimizer, dis_optimizer], train_config.lr_decay)

            # break if done training
            if cur_step == train_config.max_steps:
                break


def main():
    args = parse_arguments()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        config = OmegaConf.create(config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    munit = instantiate(config.munit)
    gen_optimizer = instantiate(config.optim, list(munit.gen_a.parameters()) + list(munit.gen_b.parameters()))
    dis_optimizer = instantiate(config.optim, list(munit.dis_a.parameters()) + list(munit.dis_b.parameters()))

    train_dataloader = instantiate(config.train_dataloader, instantiate(config.train_dataset))

    start_step = 0
    if config.resume_checkpoint is not None:
        state_dict = torch.load(config.resume_checkpoint)

        munit.load_state_dict(state_dict['munit_model_dict'])
        gen_optimizer.load_state_dict(state_dict['g_optim_dict'])
        dis_optimizer.load_state_dict(state_dict['d_optim_dict'])
        start_step = state_dict['step']
        print(f'Starting MUNIT training from checkpoints')

    else:
        print('Starting MUNIT training from random initialization')

    train(
        munit, train_dataloader,
        [gen_optimizer, dis_optimizer],
        config.train, device, start_step,
    )


if __name__ == '__main__':
    main()
