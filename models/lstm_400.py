import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from loss import positional_loss, eos_loss

torch.manual_seed(1)


class UncondHandwritingGenerator(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, num_mixture_components):
        super(UncondHandwritingGenerator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_mixture_components = num_mixture_components
        self.output_dim = 1 + 6 * self.num_mixture_components

        self.num_layers = 3
        self.lstm1 = nn.LSTM(input_dim, hidden_dim)
        self.lstm2 = nn.LSTM(input_dim, hidden_dim)
        self.lstm3 = nn.LSTM(input_dim, hidden_dim)

        self.output_transform = nn.Linear(self.hidden_dim, self.output_dim)
        self.hidden1 = self.init_hidden()
        self.hidden2 = self.init_hidden()
        self.hidden3 = self.init_hidden()

    def init_hidden(self):
        # Zero initialization of hidden state
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, hwrt_sequence):
        # Batch size fixed to 1

        seq_len = hwrt_sequence.shape[0]
        target_sequence = hwrt_sequence[1:, :]     # discard first target
        # each dim: (seq_len-1, 1)
        target_x1_seq, target_x2_seq, target_eos_seq = torch.split(
            target_sequence, 1, dim=1)

        # Input to LSTM dim: (seq_len, 1, 3)
        # lstm_output dim: (seq_len, 1, 900)
        # LSTM has skip connections
        hwrt_sequence_unsqueezed = torch.unsqueeze(hwrt_sequence, 1)
        lstm1_output, self.hidden1 = self.lstm1(
            hwrt_sequence_unsqueezed, self.hidden1)
        lstm2_input = lstm1_output + hwrt_sequence_unsqueezed
        lstm2_output, self.hidden2 = self.lstm1(lstm2_input, self.hidden2)
        lstm3_input = lstm2_output + hwrt_sequence_unsqueezed
        lstm3_output, self.hidden3 = self.lstm1(lstm3_input, self.hidden3)

        lstm_output = lstm3_output + lstm2_output + lstm1_output

        lstm_output.squeeze_()    # (seq_len, 900)
        output_sequence = self.output_transform(lstm_output)
        output_sequence = output_sequence[:-1, :]  # discard last output
        # output dim: (seq_len-1, 121)

        m = self.num_mixture_components
        e_hat = output_sequence[:, 0]
        assert output_sequence.shape[1] == (6*m+1)
        pi_hat, mu1_hat, mu2_hat, sigma1_hat, sigma2_hat, rho_hat = torch.split(
            output_sequence[:, 1:], m, dim=1)

        e = F.sigmoid(e_hat)    # (seq_len-1)
        pi = F.softmax(pi_hat, dim=0)
        mu1 = mu1_hat
        mu2 = mu2_hat
        sigma1 = torch.exp(sigma1_hat)
        sigma2 = torch.exp(sigma2_hat)
        rho = torch.tanh(rho_hat)   # each dim (seq_len-1, m)

        target_x1_seq_expanded = target_x1_seq.expand(seq_len - 1, m)
        target_x2_seq_expanded = target_x2_seq.expand(seq_len - 1, m)
        # each target dim (seq_len-1, m)
        target_eos_seq.squeeze_()

        loss_positional = positional_loss(
            (mu1, mu2, sigma1, sigma2, rho),
            (target_x1_seq_expanded, target_x2_seq_expanded),
            pi
        )
        loss_eos = eos_loss(e, target_eos_seq)
        loss = loss_positional + loss_eos
        return loss_positional, loss_eos, loss
