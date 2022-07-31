import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.nn import BatchNorm1d, Conv1d, Dropout, Flatten, Linear, Sequential, Softmax, Tanh, MaxPool1d, ReLU, Sigmoid
import math
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
# torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
# padding_mode='zeros')
from utils.gesture_data_related.read_data import read_data
from utils.neural_network_related.format_data_for_nn import format_batch_data
from utils.neural_network_related.task_generator import HandGestureTask, HandGestureDataSet, get_data_loader


# input size (N,Cin,Lin) and output (N,Cout,Lout)

class CNN1DEncoder(nn.Module):
    def __init__(self, seq_len, feature_dim):
        super(CNN1DEncoder, self).__init__()
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

            # Dropout(0.10),
            # MaxPool1d(2),

            # Conv1d(in_channels=128, out_channels=feature_dim,
            #        kernel_size=3, padding=1),
            # BatchNorm1d(feature_dim),
            # ReLU(),
            #
            # Conv1d(in_channels=feature_dim, out_channels=feature_dim,
            #        kernel_size=3, padding=1),
            # BatchNorm1d(feature_dim),
            # ReLU(),
        )
    def forward(self, input):
        # dimension of signature: (batch_size x 64 x seq_len)
        # print("encoder model input shape: ", input.shape)
        cnn_out = self.cnn_layers(input.float())
        # print("encoder model output shape: ", cnn_out.shape)

        # flat_out = self.flatten(cnn_out)
        # dense_out = self.dense_layers(flat_out)

        return cnn_out



def debug():
    data_list, _ = read_data(path='./../../../HandDataset/Abdul_New_Data', window_size=30)
    total_num_classes, data_dict = format_batch_data(data_list)
    req_num_classes = 5
    inst_per_class_train = 5
    inst_per_class_test = 2
    task = HandGestureTask(data_dict = data_dict, req_num_classes=req_num_classes,
                           train_num=inst_per_class_train, test_num=inst_per_class_test)
    trainDataLoader = get_data_loader(task=task, num_inst=inst_per_class_train, num_classes=req_num_classes, split='train')

    is_cuda = torch.cuda.is_available()
    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    device = ""
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    samples,sample_labels = trainDataLoader.__iter__().next()
    samples,sample_labels = samples.to(device).float(), sample_labels.to(device)
    encoder_model = CNN1DEncoder(seq_len=30).to(device)
    # relation_model = RelationNetwork(seq_len=30, hidden_size=10).to(device)
    # embeddings = encoder_model(samples)
    # out2 = relation_model(embeddings)
    print("done")

# debug()