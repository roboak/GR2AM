import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from learning_models.snail_network.network_modules import temporal_data_encoder, AttentionBlock, TCBlock


class SnailFewShot(nn.Module):
    def __init__(self, N, K, feature_dim, use_cuda=True):
        # N-way, K-shot
        super(SnailFewShot, self).__init__()

        self.encoder = temporal_data_encoder(seq_len=30, feature_dim = feature_dim)
        num_channels = 64 + N

        num_filters = int(math.ceil(math.log(N * K + 1, 2))) # number of dilated convolution blocks that can be applied to a sequence of length = episode_len
        self.attention1 = AttentionBlock(num_channels, 64, 32)
        num_channels += 32
        self.tc1 = TCBlock(in_channels=num_channels, seq_length=N * K + 1, filters=128)
        num_channels += num_filters * 128
        self.attention2 = AttentionBlock(num_channels, 256, 128)
        num_channels += 128
        self.tc2 = TCBlock(num_channels, N * K + 1, 128)
        num_channels += num_filters * 128
        self.attention3 = AttentionBlock(num_channels, 512, 256)
        num_channels += 256
        self.fc = nn.Linear(num_channels, N)
        self.N = N
        self.K = K
        self.use_cuda = use_cuda
    # input = sequence_len*batch_size x in_channel(63) x seq_len(30)
    # labels = sequence_len*batch_size x 5
    def forward(self, input, labels):
        x = self.encoder(input) # sequence_len*batch_size x feature_dim x 1
        x = x.squeeze(2) # remove the last dimension
        batch_size = int(labels.size()[0] / (self.N * self.K + 1))
        last_idxs = [(i + 1) * (self.N * self.K + 1) - 1 for i in range(batch_size)] # batch_size
        # one_hot encoding corresponding to test label is set tp [0, 0, 0...,0]
        if self.use_cuda:
            labels[last_idxs] = torch.Tensor(np.zeros((batch_size, labels.size()[1]))).cuda()
        else:
            labels[last_idxs] = torch.Tensor(np.zeros((batch_size, labels.size()[1])))
        x = torch.cat((x, labels), 1) # episode_len*batch_size x (num_classes_per_iter + feature_dim)
        x = x.view((batch_size, self.N * self.K + 1, -1)) # batch_size x episode_len x (num_classes_per_iter + feature_dim)
        x = self.attention1(x) # batch_size x episode_len x (num_classes_per_iter + feature_dim + att1.values_size(32))
        x = self.tc1(x) # batch_size x episode_len x (num_classes_per_iter + feature_dim + att1.values_size + filter_tc1_num(128)*math.ceil(log2(episode_len)))
        x = self.attention2(x) # batch_size x episode_len x (num_classes_per_iter + feature_dim + att1.values_size + filter_tc1_num(128)*math.ceil(log2(episode_len)) + att2.values_size(128))
        x = self.tc2(x) # batch_size x episode_len x (num_classes_per_iter + feature_dim + att1.values_size + filter_tc1_num(128)*math.ceil(log2(episode_len)) + att2.values_size(128) + filter_tc2_num(128)*math.ceil(log2(episode_len)))
        x = self.attention3(x) # -1 x -1 x _ + 256
        x = self.fc(x) # batch_size x episode_len x
        return x
