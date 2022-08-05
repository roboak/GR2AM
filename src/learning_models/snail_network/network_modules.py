import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Conv1d, Dropout, Flatten, Linear, Sequential, Softmax, Tanh, MaxPool1d, ReLU, Sigmoid

class temporal_data_encoder(nn.Module):
    def __init__(self, seq_len, feature_dim):
        super(temporal_data_encoder, self).__init__()
        # self.device = device
        self.seq_len = seq_len
        self.input_channels = 63

        self.cnn_layers = Sequential(
            # out_channel = number of filters in the CNN
            Conv1d(in_channels=self.input_channels, out_channels=64,
                   kernel_size=3,padding=1),
            BatchNorm1d(64),
            ReLU(),
            Dropout(0.15),
            MaxPool1d(2),

            Conv1d(in_channels=64, out_channels=128,
                   kernel_size=3,padding=1),
            BatchNorm1d(128),
            ReLU(),
            Dropout(0.10),
            MaxPool1d(2),

            Conv1d(in_channels=128, out_channels=256,
                   kernel_size=3, padding=1),
            BatchNorm1d(256),
            ReLU(),
            Dropout(0.10),
            MaxPool1d(2),

            Conv1d(in_channels=256, out_channels=feature_dim,
                   kernel_size=3, padding=1),
            BatchNorm1d(feature_dim),
            ReLU(),
            MaxPool1d(2),
        )
    def forward(self, input):
        cnn_out = self.cnn_layers(input.float())
        return cnn_out


class CasualConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, groups=1, bias=True):
        super(CasualConv1d, self).__init__()
        self.dilation = dilation
        padding = dilation * (kernel_size - 1)
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride,
                                padding, dilation, groups, bias)
    #     output_size=[w+2*pad-(d(k-1)+1)]/s+1
    # output_size=[i + 2*p - k - (k-1)*(d-1)]/s + 1

# FIXME: Not sure why we omit the last two columns in out
    def forward(self, input):
        # Takes something of shape (N, in_channels, T),
        # returns (N, out_channels, T)
        out = self.conv1d(input)
        return out[:, :, :-self.dilation]  # TODO: make this correct for different strides/padding


class DenseBlock(nn.Module):
    def __init__(self, in_channels, dilation, filters, kernel_size=2):
        super(DenseBlock, self).__init__()
        self.casualconv1 = CasualConv1d(in_channels, filters, kernel_size, dilation=dilation)
        self.casualconv2 = CasualConv1d(in_channels, filters, kernel_size, dilation=dilation)

# FIXME: Not sure why we have multiplied the two activation functions
    def forward(self, input):
        # input is dimensions (N, in_channels, T)
        xf = self.casualconv1(input)
        xg = self.casualconv2(input)
        activations = F.tanh(xf) * F.sigmoid(xg)  # shape: (N, filters, T)
        return torch.cat((input, activations), dim=1)


class TCBlock(nn.Module):
    def __init__(self, in_channels, seq_length, filters):
        super(TCBlock, self).__init__()
        self.dense_blocks = nn.ModuleList([DenseBlock(in_channels = in_channels + i * filters, dilation= 2 ** (i + 1), filters = filters)
                                           for i in range(int(math.ceil(math.log(seq_length, 2))))])

    def forward(self, input):
        # input is dimensions (N, T, in_channels)
        input = torch.transpose(input, 1, 2)
        for block in self.dense_blocks:
            input = block(input)
        return torch.transpose(input, 1, 2)


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, key_size, value_size):
        super(AttentionBlock, self).__init__()
        self.linear_query = nn.Linear(in_channels, key_size)
        self.linear_keys = nn.Linear(in_channels, key_size)
        self.linear_values = nn.Linear(in_channels, value_size)
        self.sqrt_key_size = math.sqrt(key_size)

    def forward(self, input):
        # input is dim (N, T, in_channels) where N is the batch_size, and T is
        # the sequence length
        mask = np.array([[1 if i > j else 0 for i in range(input.shape[1])] for j in range(input.shape[1])])
        mask = torch.ByteTensor(mask).cuda()

        # import pdb; pdb.set_trace()
        keys = self.linear_keys(input)  # shape: (N, T, key_size)
        query = self.linear_query(input)  # shape: (N, T, key_size)
        values = self.linear_values(input)  # shape: (N, T, value_size)
        temp = torch.bmm(query, torch.transpose(keys, 1, 2))  # shape: (N, T, T)
        temp.data.masked_fill_(mask, -float('inf'))
        temp = F.softmax(temp / self.sqrt_key_size,
                         dim=1)  # shape: (N, T, T), broadcasting over any slice [:, x, :], each row of the matrix
        temp = torch.bmm(temp, values)  # shape: (N, T, value_size)
        return torch.cat((input, temp), dim=2)  # shape: (N, T, in_channels + value_size)


