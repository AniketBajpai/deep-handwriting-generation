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

        self.num_layers = 3
        self.lstm1 = nn.LSTM(input_dim, hidden_dim)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim)
        self.lstm3 = nn.LSTM(hidden_dim, hidden_dim)

        self.skip_input_transform1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.skip_input_transform2 = nn.Linear(self.input_dim, self.hidden_dim)
        self.skip_output_transform1 = nn.Linear(
            self.hidden_dim, self.hidden_dim)
        self.skip_output_transform2 = nn.Linear(
            self.hidden_dim, self.hidden_dim)
        self.output_transform = nn.Linear(self.hidden_dim, self.output_dim)

        self.init_hidden()

    def init_hidden(self):
        # Zero initialization of hidden state
        h1, c1 = (torch.zeros(1, 1, self.hidden_dim),
                  torch.zeros(1, 1, self.hidden_dim))
        h2, c2 = (torch.zeros(1, 1, self.hidden_dim),
                  torch.zeros(1, 1, self.hidden_dim))
        h3, c3 = (torch.zeros(1, 1, self.hidden_dim),
                  torch.zeros(1, 1, self.hidden_dim))

        self.hidden1 = h1.to(self.device), c1.to(self.device)
        self.hidden2 = h2.to(self.device), c2.to(self.device)
        self.hidden3 = h3.to(self.device), c3.to(self.device)

    def forward(self, hwrt_sequence):
        # Batch size fixed to 1

        seq_len = hwrt_sequence.shape[0]

        # Input to LSTM dim: (seq_len, 1, 3)
        # lstm_output dim: (seq_len, 1, 900)
        # LSTM has skip connections
        hwrt_sequence_unsqueezed = torch.unsqueeze(hwrt_sequence, 1)
        lstm1_output, self.hidden1 = self.lstm1(
            hwrt_sequence_unsqueezed, self.hidden1)
        lstm2_input = lstm1_output + \
            self.skip_input_transform1(hwrt_sequence_unsqueezed)
        lstm2_output, self.hidden2 = self.lstm2(lstm2_input, self.hidden2)
        lstm3_input = lstm2_output + \
            self.skip_input_transform2(hwrt_sequence_unsqueezed)
        lstm3_output, self.hidden3 = self.lstm3(lstm3_input, self.hidden3)
        lstm_output = lstm3_output + \
            self.skip_output_transform2(lstm2_output) + \
            self.skip_output_transform1(lstm1_output)
        lstm_output = lstm_output.squeeze(dim=1)    # (seq_len, 900)

        # Gradient clippinjg of lstm hidden states
        nn.utils.clip_grad_value_(lstm1_output, config.grad_lstm_clip)
        nn.utils.clip_grad_value_(lstm2_output, config.grad_lstm_clip)
        nn.utils.clip_grad_value_(lstm3_output, config.grad_lstm_clip)
        nn.utils.clip_grad_value_(lstm_output, config.grad_lstm_clip)

        output_sequence = self.output_transform(lstm_output)
        if output_sequence.shape[0] > 1:
            output_sequence = output_sequence[:-1, :]  # discard last output
        # output dim: (seq_len-1, 121)

        # Gradient clipping of outputs
        nn.utils.clip_grad_value_(output_sequence, config.grad_output_clip)

        m = self.num_mixture_components
        e_hat = output_sequence[:, 0]
        assert output_sequence.shape[1] == (6*m+1)
        pi_hat, mu1_hat, mu2_hat, sigma1_hat, sigma2_hat, rho_hat = torch.split(
            output_sequence[:, 1:], m, dim=1)

        mdl_parameters_hat = (e_hat, pi_hat, mu1_hat,
                              mu2_hat, sigma1_hat, sigma2_hat, rho_hat)
        e, pi, mu1, mu2, sigma1, sigma2, rho = compute_mdl_parameters(
            mdl_parameters_hat)
        # e: (seq_len-1) others: (seq_len-1, m)

        return (e, pi, mu1, mu2, sigma1, sigma2, rho)


class CondHandwritingGenerator(nn.Module):

    def __init__(self, device, input_dim, hidden_dim, num_attention_mixture_components, num_mixture_components):
        super(CondHandwritingGenerator, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_attention_mixture_components = num_attention_mixture_components
        self.num_attention_output_dim = 3 * num_attention_mixture_components
        self.num_mixture_components = num_mixture_components
        self.output_dim = 1 + 6 * self.num_mixture_components

        self.num_chars = config.num_chars

        self.num_layers = 3
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim)
        self.lstm3 = nn.LSTM(hidden_dim, hidden_dim)

        self.input_transform = nn.Linear(self.input_dim, self.hidden_dim)
        self.skip_input_transform2 = nn.Linear(self.input_dim, self.hidden_dim)
        self.skip_input_transform3 = nn.Linear(self.input_dim, self.hidden_dim)
        self.w_transform1 = nn.Linear(self.num_chars, self.hidden_dim)
        self.w_transform2 = nn.Linear(self.num_chars, self.hidden_dim)
        self.w_transform3 = nn.Linear(self.num_chars, self.hidden_dim)
        self.skip_output_transform1 = nn.Linear(
            self.hidden_dim, self.hidden_dim)
        self.skip_output_transform2 = nn.Linear(
            self.hidden_dim, self.hidden_dim)
        self.attention_transform = nn.Linear(
            self.hidden_dim, self.num_attention_output_dim)
        self.output_transform = nn.Linear(self.hidden_dim, self.output_dim)

        self.init_hidden()

    def init_hidden(self):
        # Zero initialization of hidden state
        h1, c1 = (torch.zeros(1, 1, self.hidden_dim),
                  torch.zeros(1, 1, self.hidden_dim))
        h2, c2 = (torch.zeros(1, 1, self.hidden_dim),
                  torch.zeros(1, 1, self.hidden_dim))
        h3, c3 = (torch.zeros(1, 1, self.hidden_dim),
                  torch.zeros(1, 1, self.hidden_dim))

        self.hidden1 = h1.to(self.device), c1.to(self.device)
        self.hidden2 = h2.to(self.device), c2.to(self.device)
        self.hidden3 = h3.to(self.device), c3.to(self.device)

    def forward(self, hwrt_sequence, char_sequence):
        # Batch size fixed to 1

        seq_len = hwrt_sequence.shape[0]
        kappa_init = torch.zeros(
            [self.num_attention_mixture_components]).to(self.device)
        w_prev = torch.zeros([seq_len, 1, self.num_chars]).to(self.device)

        target_sequence = hwrt_sequence[1:, :]     # discard first target
        # each dim: (seq_len-1, 1)
        target_eos_seq, target_x1_seq, target_x2_seq = torch.split(
            target_sequence, 1, dim=1)

        # Input to LSTM dim: (seq_len, 1, 3)
        # lstm_output dim: (seq_len, 1, 900)
        # LSTM has skip connections
        hwrt_sequence_unsqueezed = torch.unsqueeze(hwrt_sequence, 1)
        lstm_input = self.input_transform(
            hwrt_sequence_unsqueezed) + self.w_transform1(w_prev)
        lstm1_output, self.hidden1 = self.lstm1(lstm_input, self.hidden1)

        # Attention mechanism
        p = self.attention_transform(lstm1_output.squeeze(dim=1))     # (seq_len, 3K)
        alpha_hat, beta_hat, kappa_hat = torch.split(
            p, self.num_attention_mixture_components, dim=1)
        alpha = torch.exp(alpha_hat)
        beta = torch.exp(beta_hat)
        kappa = torch.cumsum(torch.exp(kappa_hat), dim=1)
        # each (seq_len, K)

        phi_t_u_list = []
        U = char_sequence.shape[0]
        for u in range(U):
            phi = torch.sum(alpha * torch.exp(-1 * beta *
                                              (kappa - u)**2), dim=1)   # (seq_len)
            phi_t_u_list.append(phi)
        phi_t_u = torch.stack(phi_t_u_list, dim=1)  # (seq,len, U)
        w = torch.mm(phi_t_u, char_sequence)        # (seq,len, num_chars)
        w = torch.unsqueeze(w, 1)                   # (seq,len, 1, num_chars)
        w_prev = w

        lstm2_input = lstm1_output + \
            self.skip_input_transform2(
                hwrt_sequence_unsqueezed) + self.w_transform2(w)
        lstm2_output, self.hidden2 = self.lstm2(lstm2_input, self.hidden2)

        lstm3_input = lstm2_output + \
            self.skip_input_transform3(
                hwrt_sequence_unsqueezed) + self.w_transform3(w)
        lstm3_output, self.hidden3 = self.lstm3(lstm3_input, self.hidden3)

        lstm_output = lstm3_output + \
            self.skip_output_transform2(lstm2_output) + \
            self.skip_output_transform1(lstm1_output)
        lstm_output = lstm_output.squeeze(dim=1)    # (seq_len, 900)

        # Gradient clippinjg of lstm hidden states
        nn.utils.clip_grad_value_(lstm1_output, config.grad_lstm_clip)
        nn.utils.clip_grad_value_(lstm2_output, config.grad_lstm_clip)
        nn.utils.clip_grad_value_(lstm3_output, config.grad_lstm_clip)
        nn.utils.clip_grad_value_(lstm_output, config.grad_lstm_clip)

        output_sequence = self.output_transform(lstm_output)
        output_sequence = output_sequence[:-1, :]  # discard last output
        # output dim: (seq_len-1, 121)

        # Gradient clipping of outputs
        nn.utils.clip_grad_value_(output_sequence, config.grad_output_clip)

        m = self.num_mixture_components
        e_hat = output_sequence[:, 0]
        assert output_sequence.shape[1] == (6*m+1)
        pi_hat, mu1_hat, mu2_hat, sigma1_hat, sigma2_hat, rho_hat = torch.split(
            output_sequence[:, 1:], m, dim=1)

        e = torch.sigmoid(e_hat)    # (seq_len-1)
        pi = F.softmax(pi_hat, dim=0)

        target_x1_seq_expanded = target_x1_seq.expand(seq_len - 1, m)
        target_x2_seq_expanded = target_x2_seq.expand(seq_len - 1, m)
        # each target dim (seq_len-1, m)
        target_eos_seq = target_eos_seq.squeeze()

        loss_positional = positional_loss(
            (mu1_hat, mu2_hat, sigma1_hat, sigma2_hat, rho_hat),
            (target_x1_seq_expanded, target_x2_seq_expanded),
            pi
        )
        loss_eos = eos_loss(e, target_eos_seq)
        loss = loss_positional + loss_eos

        return loss_positional, loss_eos, loss
