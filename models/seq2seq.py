import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config

torch.manual_seed(1)


class HandwritingRecognizer(nn.Module):
    """ seq2seq with attention module """

    def __init__(self, device, input_dim, hidden_dim, output_dim):
        super(HandwritingRecognizer, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_stroke_length = config.max_stroke_length
        self.output_dim = output_dim

        self.num_layers = 1
        self.encoder_lstm = nn.LSTM(self.input_dim, self.hidden_dim)

        self.attention = nn.Linear(
            self.hidden_dim + self.output_dim, self.max_stroke_length)
        self.attention_combine = nn.Linear(
            self.hidden_dim + self.output_dim, self.output_dim)

        self.decoder_lstm_cell = nn.LSTMCell(self.output_dim, self.hidden_dim)

        self.output_transform = nn.Linear(self.hidden_dim, self.output_dim)

        self.init_hidden()

    def init_hidden(self):
        # Zero initialization of hidden state
        e_h0, e_c0 = (torch.zeros(1, 1, self.hidden_dim),
                      torch.zeros(1, 1, self.hidden_dim))
        self.encoder_hidden = e_h0.to(self.device), e_c0.to(self.device)
        d_h0, d_c0 = (torch.zeros(1, self.hidden_dim),
                      torch.zeros(1, self.hidden_dim))
        self.decoder_hidden = d_h0.to(self.device), d_c0.to(self.device)

    def forward(self, hwrt_sequence, char_sequence):
        # Batch size fixed to 1

        hwrt_seq_len = hwrt_sequence.shape[0]
        char_seq_len = char_sequence.shape[0]

        # Encoder
        # Input to LSTM dim: (hwrt_seq_len, 1, num_chars)
        # encoder_output dim: (hwrt_seq_len, 1, hidden_dim)
        encoder_outputs, self.encoder_hidden = self.encoder_lstm(
            torch.unsqueeze(hwrt_sequence, 1), self.encoder_hidden)
        encoder_outputs = encoder_outputs.squeeze(
            dim=1)    # (hwrt_seq_len, hidden_dim)

        # Gradient clipping of lstm hidden states
        nn.utils.clip_grad_value_(encoder_outputs, config.grad_lstm_clip)

        char_sequence_unsqueezed = torch.unsqueeze(char_sequence, 1)
        decoder_output_sequence = []
        # Encoder hidden state passed to decoder
        self.decoder_hidden = (
            self.encoder_hidden[0][0], self.encoder_hidden[1][0])

        for t in range(char_seq_len-1):
            current_char = char_sequence_unsqueezed[t]      # (1, num_chars)
            # Attention
            attention_input = torch.cat(
                (current_char, self.decoder_hidden[0]), dim=1)  # (1, output_dim + hidden_dim)
            attention_weights = F.softmax(self.attention(
                attention_input), dim=1)  # (1, max_stroke_length)
            attention_applied = torch.mm(
                attention_weights[:, :hwrt_seq_len], encoder_outputs)     # (1, hidden_dim)

            attention_out = torch.cat((current_char, attention_applied), 1)
            decoder_input = F.relu(self.attention_combine(
                attention_out))   # (1, num_chars)

            # Decoder
            self.decoder_hidden = self.decoder_lstm_cell(
                decoder_input, self.decoder_hidden)    # ((1, hidden_dim), (1, hidden_dim))
            decoder_output = self.output_transform(self.decoder_hidden[0])   # (1, num_chars)
            decoder_output = F.softmax(decoder_output, dim=1)
            decoder_output = decoder_output.squeeze()
            # Gradient clipping of outputs
            nn.utils.clip_grad_value_(decoder_output, config.grad_output_clip)
            decoder_output_sequence.append(decoder_output)  # Output logits

        decoder_output_sequence = torch.stack(decoder_output_sequence)
        return decoder_output_sequence
