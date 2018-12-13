import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import logging
import tensorboard_logger as tb_log

import models.lstm_400 as lstm_400
import models.lstm_900 as lstm_900
from dataloader import DataLoader
import config

name = 'unconditional_900'

os.makedirs('./runs', exist_ok=True)
os.makedirs('./runs/{}'.format(name), exist_ok=True)
os.makedirs('./logs', exist_ok=True)
os.makedirs('./logs/{}'.format(name), exist_ok=True)
os.makedirs('./checkpoints', exist_ok=True)
os.makedirs('./checkpoints/{}'.format(name), exist_ok=True)
logger = logging.getLogger('train_unconditional')
logging.basicConfig(
    filename='logs/{}/debug.log'.format(name), level=logging.DEBUG)
logging.basicConfig(
    filename='logs/{}/info.log'.format(name), level=logging.INFO)
# logger.setLevel(logging.INFO)
tb_log.configure('runs/{}'.format(name), flush_secs=5)

dataloader = DataLoader()
model = lstm_900.UncondHandwritingGenerator(
    config.INPUT_DIM, config.HIDDEN_DIM_900, config.num_mixture_components)
optimizer = optim.RMSprop(model.parameters(), lr=config.lr,
                          alpha=config.alpha, momentum=config.momentum)
model.train()   # model in training mode

for epoch in range(config.MAX_EPOCHS):
    for i, hwrt_sequences in enumerate(dataloader.get_minibatch()):
        start_time = time.time()

        # Clear gradients and initialize hidden at the start of each minibatch
        model.zero_grad()
        model.hidden = model.init_hidden()

        # forward pass
        loss_positional, loss_eos, loss = model(hwrt_sequences)
        # TODO: Check if normalize required: divide by max timestep, batch size

        # backward pass
        loss.backward()
        optimizer.step()

        end_time = time.time()
        seq_length = hwrt_sequences.shape[0]    # Assumption: batch size 1
        time_elapsed = end_time - start_time
        # logger.debug('Time per iteration: {} Sequence length: {}'.format(
        #     time_elapsed, seq_length))

        # periodically write loss
        if i % config.log_freq == 0:
            logger.debug('Itertion {}: Loss- Total: {} Positional: {} EOS: {}'.format(i, loss_positional, loss_eos, loss))
            tb_log.log_value('loss_positional', loss_positional, i)
            tb_log.log_value('loss_eos', loss_eos, i)
            tb_log.log_value('loss_', loss, i)
            
    # Plot statistics after epoch
    logger.info('Epoch {}: Loss- Total: {} Positional: {} EOS: {}'.format(epoch, loss_positional, loss_eos, loss))
    tb_log.log_value('epoch_loss_positional', loss_positional, epoch)
    tb_log.log_value('epoch_loss_eos', loss_eos, epoch)
    tb_log.log_value('epoch_loss_', loss, epoch)

    # Save checkpoint
    ckpt_path = 'checkpoints/{}/{}.pth'.format(name, epoch)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, ckpt_path)
