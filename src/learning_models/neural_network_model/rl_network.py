import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.nn import BatchNorm1d, Conv1d, Dropout, Flatten, Linear, Sequential, Softmax, Tanh, MaxPool1d, ReLU, Sigmoid
import math

# from torch.utils.tensorboard import SummaryWriter
# torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
# padding_mode='zeros')
from utils.gesture_data_related.read_data import read_data
from utils.neural_network_related.format_data_for_nn import format_batch_data
from utils.neural_network_related.task_generator import HandGestureTask, HandGestureDataSet, get_data_loader

parser = argparse.ArgumentParser(description="One Shot Gesture Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 64)
parser.add_argument("-r","--relation_dim",type = int, default = 8)
parser.add_argument("-w","--class_num",type = int, default = 5)
parser.add_argument("-s","--sample_num_per_class",type = int, default = 5)
parser.add_argument("-b","--batch_num_per_class",type = int, default = 15)
parser.add_argument("-e","--episode",type = int, default= 1000000)
parser.add_argument("-t","--test_episode", type = int, default = 1000)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
args = parser.parse_args()


# Hyper Parameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit

# input size (N,Cin,Lin) and output (N,Cout,Lout)

class CNN1DEncoder(nn.Module):
    def __init__(self, seq_len):
        super(CNN1DEncoder, self).__init__()
        # self.device = device
        self.seq_len = seq_len
        self.input_channels = 63
        self.n_cnn_filter_1 = 128
        self.n_cnn_filter_2 = 64
        self.cnn_kernel_size_1 = 1
        self.cnn_kernel_size_2 = 1
        self.cnn_stride_1 = 1
        self.cnn_stride_2 = 1

        self.cnn_layers = Sequential(
            # out_channel = number of filters in the CNN
            Conv1d(in_channels=self.input_channels, out_channels=self.n_cnn_filter_1,
                   kernel_size=3,padding=1),
            BatchNorm1d(self.n_cnn_filter_1),
            Tanh(),
            Dropout(0.15),
            # MaxPool1d(2),
            Conv1d(in_channels=self.n_cnn_filter_1, out_channels=self.n_cnn_filter_2,
                   kernel_size=3,padding=1),
            BatchNorm1d(self.n_cnn_filter_2),
            Tanh(),
            Dropout(0.10),
            # MaxPool1d(2),
        )
    def forward(self, input):
        # dimension of signature: (batch_size x 64 x seq_len)
        print("encoder model input shape: ", input.shape)
        cnn_out = self.cnn_layers(input)
        print("encoder model output shape: ", cnn_out.shape)

        # flat_out = self.flatten(cnn_out)
        # dense_out = self.dense_layers(flat_out)

        return cnn_out

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self, seq_len: int, hidden_size: int):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv1d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm1d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool1d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv1d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm1d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool1d(2))
        # batch_size x 64 x seq_len/4
        # input_szie  = 64 x seq_len/4
        input_size = 448
        self.layer3 = Sequential(
            nn.Linear(input_size, hidden_size),
            ReLU(),
            nn.Linear(hidden_size, 1),
            Sigmoid()
        )
        # self.fc1 = nn.Linear(input_size,hidden_size)
        # self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x):
        print('cnn input shape {}'.format(x.shape))
        out = self.layer1(x)
        print('cnn 1 output shape {}'.format(out.shape))
        out = self.layer2(out)
        print('cnn 2 output shape {}'.format(out.shape))
        out = out.view(out.size(0),-1)
        print('cnn reshaped output shape {}'.format(out.shape))
        # out = ReLU(self.fc1(out))
        # out = Sigmoid(self.fc2(out))
        out = self.layer3(out)
        print('fc layer output shape {}'.format(out.shape))

        return out




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
    relation_model = RelationNetwork(seq_len=30, hidden_size=10).to(device)
    embeddings = encoder_model(samples)
    out2 = relation_model(embeddings)
    print("done")

debug()