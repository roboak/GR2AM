import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from read_data import read_data

device = torch.device("cpu")
sequence_length = 21
batch_size = 3


# We want to input each capture row by row, i.e. each frame into the model, thus we have as the input the 21x3 matrix
# which we possibly need to reshape/flatten into 63x1, as X Y Z are all important for each landmark
class LSTMNetwork(nn.Module):
    def __init__(self, input_size=21, output_size=3, hidden_layer_size=10, drop_prob=0.2):
        super(LSTMNetwork, self).__init__()
        self.output_size = output_size
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, bidirectional=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_layer_size, output_size)
        self.activation = nn.Softmax()
        # (h0, c0) => num_layers, input_size, hidden_size
        self.hidden_cell = (torch.zeros(1, 21, self.hidden_layer_size), torch.zeros(1, 21, self.hidden_layer_size))

    def forward(self, input_seq):
        input_seq = torch.Tensor(input_seq)
        input_seq = input_seq.view(len(input_seq), batch_size, -1)  # ✅ Here we now got a tensor containing 21 pairs of 3, so X Y Z

        lstm_out, _ = self.lstm(input_seq, self.hidden_cell)

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_layer_size)
        out = self.dropout(lstm_out)
        out = self.activation(out)
        out = self.fc(out)
        return out


# In the case of multiclass classification use loss_function = nn.CrossEntropyLoss()
model = LSTMNetwork(hidden_layer_size=10, drop_prob=0.2)
model = model.float()
optimizer = torch.optim.Adamax(model.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()
gesture_data = read_data()
# Train the model
epochs = 100
single_loss = None

for i in range(epochs):
    for file in gesture_data:  # for each capture
        seq, labels = file.data, file.label

        for frame in seq:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 3, model.hidden_layer_size), torch.zeros(1, 3, model.hidden_layer_size))
            y_pred = model(frame)
            # labels = torch.reshape(labels, -1)
            print(y_pred.shape)

            # FIXME below should rather be indexing the probability
            ###
            x = int(labels)
            labels = torch.empty(1, 3)  # 3 = input_size  -> tensor([0., 0., 0.])
            labels[0][x] = 1.
            labels = torch.tensor([0.0, 1.0, 0.0])  # -> tensor([0., 1., 0.])
            print(labels)
            ###

            # gesture1 -> tensor([1., 0., 0.])
            # gesture2 -> tensor([0., 1., 0.])
            # gesture3 -> tensor([0., 0., 1.])

            single_loss = loss_function(y_pred, labels)  # pred = no classes, target is simply the index
            single_loss.backward()
            optimizer.step()
    if i % 10 == 1:
        print(f'epoch: {i} loss: {single_loss.item()}')

"""
# Testing the model
model.eval()
Original_Labels = []
Predicted_Labels = []

for seq, labels in Test_Data_Loader:
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        # Predicted_Labels.append(model(seq.float()).item())
        dummy = model(seq.float())
        dummy = torch.round(dummy)
        Predicted_Labels.append(dummy.item())
        Original_Labels.append(labels.float())
"""
