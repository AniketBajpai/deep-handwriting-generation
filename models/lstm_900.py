import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from utils import compute_mdl_parameters

torch.manual_seed(1)


class UncondHandwritingGenerator(nn.Module):

    def __init__(self, device, input_dim, hidden_dim, num_mixture_components):
        super(UncondHandwritingGenerator, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_mixture_components = num_mixture_components
        self.output_dim = 1 + 6 * self.num_mixture_components

        self.num_layers = 1
        self.lstm = nn.LSTM(input_dim, hidden_dim)

        self.output_transform = nn.Linear(
            self.hidden_dim, self.output_dim)     # Initialization from paper?

        self.init_hidden()

    def init_hidden(self):
        # Zero initialization of hidden state
        h0, c0 = (torch.zeros(1, 1, self.hidden_dim),
                  torch.zeros(1, 1, self.hidden_dim))
        self.hidden = h0.to(self.device), c0.to(self.device)

    def forward(self, hwrt_sequence):
        # Batch size fixed to 1

        seq_len = hwrt_sequence.shape[0]

        # Input to LSTM dim: (seq_len, 1, 3)
        # lstm_output dim: (seq_len, 1, 900)
        lstm_output, self.hidden = self.lstm(
            torch.unsqueeze(hwrt_sequence, 1), self.hidden)
        lstm_output = lstm_output.squeeze(dim=1)    # (seq_len, 900)
        # Gradient clippinjg of lstm hidden states
        nn.utils.clip_grad_value_(lstm_output, config.grad_lstm_clip)

        output_sequence = self.output_transform(lstm_output)
        if output_sequence.shape[0] > 1:
            output_sequence = output_sequence[:-1, :]  # discard last output
        # output dim: (seq_len-1, 121)

        # Gradient clipping of outputs
        nn.utils.clip_grad_value_(output_sequence, config.grad_output_clip)

        m = self.num_mixture_components
        e_hat = output_sequence[:, 0]
        assert output_sequence.shape[1] == (6 * m + 1)
        pi_hat, mu1_hat, mu2_hat, sigma1_hat, sigma2_hat, rho_hat = torch.split(
            output_sequence[:, 1:], m, dim=1)
        # each dim: (seq_len-1, m)

        mdl_parameters_hat = (e_hat, pi_hat, mu1_hat,
                              mu2_hat, sigma1_hat, sigma2_hat, rho_hat)
        e, pi, mu1, mu2, sigma1, sigma2, rho = compute_mdl_parameters(
            mdl_parameters_hat)
        # e: (seq_len-1) others: (seq_len-1, m)

        return (e, pi, mu1, mu2, sigma1, sigma2, rho)
