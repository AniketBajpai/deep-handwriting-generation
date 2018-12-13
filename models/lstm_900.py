import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from loss import positional_loss, eos_loss

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

        self.output_transform = nn.Linear(self.hidden_dim, self.output_dim)     # Initiazation from paper?
        
        self.init_hidden()

    def init_hidden(self):
        # Zero initialization of hidden state
        h0, c0 = (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))
        self.hidden = h0.to(self.device), c0.to(self.device)

    def forward(self, hwrt_sequence):
        # Batch size fixed to 1

        seq_len = hwrt_sequence.shape[0]
        target_sequence = hwrt_sequence[1:, :]     # discard first target
        # each dim: (seq_len-1, 1)
        target_eos_seq, target_x1_seq, target_x2_seq = torch.split(
            target_sequence, 1, dim=1)

        # Input to LSTM dim: (seq_len, 1, 3)
        # lstm_output dim: (seq_len, 1, 900)
        lstm_output, self.hidden = self.lstm(
            torch.unsqueeze(hwrt_sequence, 1), self.hidden)
        lstm_output = lstm_output.squeeze()    # (seq_len, 900)
        
        # Gradient clippinjg of lstm hidden states
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

        # loss_positional = loss_positional / seq_len
        # loss_eos = loss_eos / seq_len

        loss = loss_positional + loss_eos

        return loss_positional, loss_eos, loss
