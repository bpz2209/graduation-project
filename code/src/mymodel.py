'''
Description: 
Version: 1.0
Autor: Julian Lin
Date: 2023-01-09 20:54:13
LastEditors: Julian Lin
LastEditTime: 2023-01-10 13:38:53
'''
import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size,
                          num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_size).to(x.device)

        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out


class CNN(nn.Module):
    def __init__(self, input_size, output_size, num_filters=32, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, num_filters, kernel_size, stride)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(num_filters, output_size)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out


class Transformer(nn.Module):
    def __init__(self, input_size, output_size, d_model=64, nhead=2, num_layers=2, dropout=0.1):
        super().__init__()
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(input_size, dropout)
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=256, dropout=dropout, activation='relu')
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, num_layers)
        self.encoder = nn.Linear(input_size, d_model)
        self.decoder = nn.Linear(d_model, output_size)
        self.output_size = output_size

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.output_size)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output[:, -1, :])
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask
