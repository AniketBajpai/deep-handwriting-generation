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

import models.lstm_multilayer as lstm_multilayer
import models.lstm_singlelayer as lstm_singlelayer
import models.seq2seq as seq2seq
from dataloader import DataLoader
from loss import positional_loss, eos_loss
import config

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="name of run")
parser.add_argument("--model", help="type of model")
parser.add_argument("--reg", default=None, help="type of regularization")
parser.add_argument("--use_gpu", action='store_true',
                    help="flag to check whether to use gpu")
parser.add_argument("--ckpt", help="path from checkpoint should be loaded")
args = parser.parse_args()

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

if args.model == 'generation_unconditional_900':
    task_type = 'gen'
    is_conditional = False
    model = lstm_singlelayer.UncondHandwritingGenerator(
        device, config.INPUT_DIM, config.HIDDEN_DIM_900, config.num_mixture_components)
elif args.model == 'generation_unconditional_400':
    task_type = 'gen'
    is_conditional = False
    model = lstm_multilayer.UncondHandwritingGenerator(
        device, config.INPUT_DIM, config.HIDDEN_DIM_400, config.num_mixture_components)
elif args.model == 'generation_conditional_400':
    task_type = 'gen'
    is_conditional = True
    model = lstm_multilayer.CondHandwritingGenerator(
        device, config.INPUT_DIM, config.HIDDEN_DIM_400, config.K, config.num_mixture_components)
elif args.model == 'recognition':
    task_type = 'rec'
    is_conditional = True
    num_outputs = config.num_chars + 3  # <sos>, <eos>, space
    model = seq2seq.HandwritingRecognizer(
        device, config.INPUT_DIM, config.HIDDEN_DIM_REC, num_outputs)

if args.reg is None:
    optimizer = optim.RMSprop(model.parameters(), lr=config.lr, eps=config.epsilon,
                              alpha=config.alpha, momentum=config.momentum)
elif args.reg == 'l2':
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


def forward_generation(model, data, device, is_conditional):
    # Clear gradients and initialize hidden at the start of each minibatch
    model.zero_grad()
    model.init_hidden()

    if is_conditional:
        hwrt_sequences, char_sequences = data
        hwrt_sequences, char_sequences = hwrt_sequences.to(
            device), char_sequences.to(device)
    else:
        hwrt_sequences = data.to(device)

    # seq_length = hwrt_sequences.shape[0]    # Assumption: batch size 1

    # Forward pass
    if is_conditional:
        mdl_parameters = model(hwrt_sequences, char_sequences)
    else:
        mdl_parameters = model(hwrt_sequences)

    # Calculate loss
    e, pi, mu1, mu2, sigma1, sigma2, rho = mdl_parameters
    seq_len = hwrt_sequences.shape[0]
    m = config.num_mixture_components
    target_sequence = hwrt_sequences[1:, :]     # discard first target
    # each dim: (seq_len-1, 1)
    target_eos_seq, target_x1_seq, target_x2_seq = torch.split(
        target_sequence, 1, dim=1)
    target_x1_seq_expanded = target_x1_seq.expand(seq_len - 1, m)
    target_x2_seq_expanded = target_x2_seq.expand(seq_len - 1, m)
    # each target dim (seq_len-1, m)
    target_eos_seq = target_eos_seq.squeeze(dim=1)

    loss_positional = positional_loss(
        (mu1, mu2, sigma1, sigma2, rho),
        (target_x1_seq_expanded, target_x2_seq_expanded),
        pi
    )
    loss_eos = eos_loss(e, target_eos_seq)
    loss = loss_positional + loss_eos
    return loss, loss_positional, loss_eos


def forward_recognition(model, data, device):
    # Clear gradients and initialize hidden at the start of each minibatch
    model.zero_grad()
    model.init_hidden()

    hwrt_sequences, char_sequences = data
    hwrt_sequences, char_sequences = hwrt_sequences.to(
        device), char_sequences.to(device)
    
    # Forward pass
    decoder_outputs = model(hwrt_sequences, char_sequences)

    # Calculate loss
    criterion = nn.BCELoss()    # serves as cross entropy when targets are one-hot
    target_sequence = char_sequences[1:, :]
    loss = criterion(decoder_outputs, target_sequence)
    
    return loss


for epoch in range(start_epoch, config.MAX_EPOCHS):
    for i, data in enumerate(dataloader.get_minibatch(task_type, is_conditional)):
        start_time = time.time()

        # Forward pass
        if task_type == 'gen':
            loss, loss_positional, loss_eos = forward_generation(
                model, data, device, is_conditional)
        elif task_type == 'rec':
            loss = forward_recognition(model, data, device)

        # backward pass
        loss.backward()
        # Clip gradients for all parameters
        nn.utils.clip_grad_value_(model.parameters(), 1)
        optimizer.step()

        end_time = time.time()

        time_elapsed = end_time - start_time
        # logger.debug('Time per iteration: {} Sequence length: {}'.format(
        #     time_elapsed, seq_length))

        # periodically write loss
        if i % config.log_freq == 0:
            if task_type == 'gen':
                logger.debug('Iteration {}: Loss:- Total: {} Positional: {} EOS: {}'.format(
                    i, loss, loss_positional, loss_eos))
            elif task_type == 'rec':
                logger.debug('Iteration {}: Loss:- Total: {}'.format(i, loss))
            counter = config.MAX_EPOCHS * config.examples_per_epoch + i
            # tb_log.log_value('loss_positional', loss_positional, counter)
            # tb_log.log_value('loss_eos', loss_eos, counter)
            # tb_log.log_value('loss', loss, counter)

    # Plot statistics after epoch
    random_train_examples = dataloader.get_random_examples(
        config.num_test_examples, 'train', task_type, is_conditional)
    random_test_examples = dataloader.get_random_examples(
        config.num_test_examples, 'test', task_type, is_conditional)

    train_loss = 0
    train_loss_positional = 0
    train_loss_eos = 0

    for data in random_train_examples:
        if task_type == 'gen':
            loss, loss_positional, loss_eos = forward_generation(
                model, data, device, is_conditional)
            train_loss_positional += loss_positional
            train_loss_eos += loss_eos
        elif task_type == 'rec':
            loss = forward_recognition(model, data, device)
        train_loss += loss

    if task_type == 'gen':
        logger.info('Epoch {}: Train loss- Total: {} Positional: {} EOS: {}'.format(epoch,
                                                                            train_loss, train_loss_positional, train_loss_eos))
        tb_log.log_value('epoch_train_loss_positional',
                        train_loss_positional, epoch)
        tb_log.log_value('epoch_train_loss_eos', train_loss_eos, epoch)
    elif task_type == 'rec':
        logger.info('Epoch {}: Train loss- Total: {}'.format(epoch, train_loss))
        
    tb_log.log_value('epoch_train_loss', train_loss, epoch)

    test_loss = 0
    test_loss_positional = 0
    test_loss_eos = 0

    for data in random_test_examples:
        if task_type == 'gen':
            loss, loss_positional, loss_eos = forward_generation(
                model, data, device, is_conditional)
            test_loss_positional += loss_positional
            test_loss_eos += loss_eos
        elif task_type == 'rec':
            loss = forward_recognition(model, data, device)
        test_loss += loss

    if task_type == 'gen':
        logger.info('Epoch {}: Test loss- Total: {} Positional: {} EOS: {}'.format(epoch,
                                                                                    test_loss, test_loss_positional, test_loss_eos))
        tb_log.log_value('epoch_test_loss_positional',
                         test_loss_positional, epoch)
        tb_log.log_value('epoch_test_loss_eos', test_loss_eos, epoch)
    elif task_type == 'rec':
        logger.info('Epoch {}: Test loss- Total: {}'.format(epoch, test_loss))

    tb_log.log_value('epoch_test_loss', test_loss, epoch)

    # Save checkpoint
    ckpt_path = 'checkpoints/{}/{}.pth'.format(name, epoch)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, ckpt_path)
