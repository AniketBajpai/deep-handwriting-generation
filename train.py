import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import logging
import tensorboard_logger as tb_log
import argparse

import models.lstm_400 as lstm_400
import models.lstm_900 as lstm_900
from dataloader import DataLoader
import config

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="name of run")
parser.add_argument("--model", help="type of model")
parser.add_argument("--reg", default=None, help="type of regularization")
parser.add_argument("--use_gpu", action='store_true',
                    help="flag to check whether to use gpu")
parser.add_argument("--ckpt", help="path from checkpoint should be loaded")
args = parser.parse_args()

# name = 'unconditional_900'
name = args.name
check_gpu = args.use_gpu and torch.cuda.is_available()
device = torch.device('cuda' if check_gpu else 'cpu')

os.makedirs('./runs', exist_ok=True)
os.makedirs('./runs/{}'.format(name), exist_ok=True)
os.makedirs('./logs', exist_ok=True)
os.makedirs('./logs/{}'.format(name), exist_ok=True)
os.makedirs('./checkpoints', exist_ok=True)
os.makedirs('./checkpoints/{}'.format(name), exist_ok=True)
logger = logging.getLogger('train')
logging.basicConfig(
    filename='logs/{}/debug.log'.format(name), level=logging.DEBUG)
logging.basicConfig(
    filename='logs/{}/info.log'.format(name), level=logging.INFO)
# logger.setLevel(logging.INFO)
tb_log.configure('runs/{}'.format(name), flush_secs=5)

dataloader = DataLoader()

if args.model == 'unconditional_900':
    is_conditional = False
    model = lstm_900.UncondHandwritingGenerator(
        device, config.INPUT_DIM, config.HIDDEN_DIM_900, config.num_mixture_components)
elif args.model == 'unconditional_400':
    is_conditional = False
    model = lstm_400.UncondHandwritingGenerator(
        device, config.INPUT_DIM, config.HIDDEN_DIM_400, config.num_mixture_components)
elif args.model == 'conditional_400':
    is_conditional = True
    model = lstm_400.CondHandwritingGenerator(
        device, config.INPUT_DIM, config.HIDDEN_DIM_400, config.K, config.num_mixture_components)

if args.reg is None:
    optimizer = optim.RMSprop(model.parameters(), lr=config.lr, eps=config.epsilon,
                              alpha=config.alpha, momentum=config.momentum)
elif args.reg is 'l2':
    optimizer = optim.RMSprop(model.parameters(), lr=config.lr, eps=config.epsilon,
                              alpha=config.alpha, momentum=config.momentum,
                              weight_decay=config.l2_reg_lambda)

model.train()   # model in training mode
model.to(device)

if args.ckpt:
    checkpoint = torch.load(args.ckpt)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = epoch = checkpoint['epoch']
else:
    start_epoch = 0

for epoch in range(start_epoch, config.MAX_EPOCHS):
    for i, data in enumerate(dataloader.get_minibatch(is_conditional)):
        start_time = time.time()

        if is_conditional:
            hwrt_sequences, char_sequences = data
            hwrt_sequences, char_sequences = hwrt_sequences.to(
                device), char_sequences.to(device)
        else:
            hwrt_sequences = data.to(device)

        # Clear gradients and initialize hidden at the start of each minibatch
        model.zero_grad()
        model.init_hidden()

        # forward pass
        if is_conditional:
            loss_positional, loss_eos, loss = model(
                hwrt_sequences, char_sequences)
        else:
            loss_positional, loss_eos, loss = model(hwrt_sequences)

        # backward pass
        loss.backward()
        # Clip gradients for all parameters
        nn.utils.clip_grad_value_(model.parameters(), 1)
        optimizer.step()

        end_time = time.time()
        seq_length = hwrt_sequences.shape[0]    # Assumption: batch size 1
        time_elapsed = end_time - start_time
        # logger.debug('Time per iteration: {} Sequence length: {}'.format(
        #     time_elapsed, seq_length))

        # periodically write loss
        if i % config.log_freq == 0:
            logger.debug('Iteration {}: Loss- Total: {} Positional: {} EOS: {}'.format(
                i, loss_positional, loss_eos, loss))
            tb_log.log_value('loss_positional', loss_positional, i)
            tb_log.log_value('loss_eos', loss_eos, i)
            tb_log.log_value('loss', loss, i)

    # Plot statistics after epoch
    logger.info('Epoch {}: Loss- Total: {} Positional: {} EOS: {}'.format(epoch,
                                                                          loss_positional, loss_eos, loss))
    tb_log.log_value('epoch_loss_positional', loss_positional, epoch)
    tb_log.log_value('epoch_loss_eos', loss_eos, epoch)
    tb_log.log_value('epoch_loss', loss, epoch)

    # Save checkpoint
    ckpt_path = 'checkpoints/{}/{}.pth'.format(name, epoch)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, ckpt_path)
