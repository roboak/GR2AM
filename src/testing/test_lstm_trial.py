import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

device = torch.device("cpu")
sequence_length = 21
batch_size = 1

"""
LSTM algorithm accepts three inputs: previous hidden state, previous cell state and current input. 
The hidden_cell variable contains the previous hidden and cell state. 
The lstm and linear layer variables are used to create the LSTM and linear layers.
"""


class LSTMNetwork(nn.Module):
    def __init__(self, input_size=21, output_size=1, hidden_layer_size=10, drop_prob=0.5):
        super(LSTMNetwork, self).__init__()
        self.output_size = output_size
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.dropout = nn.Dropout(drop_prob)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.activation = nn.Sigmoid()
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size), torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_layer_size)
        out = self.dropout(lstm_out)
        out = self.linear(out)
        out = self.activation(out)
        out = out.reshape(-1)
        # out = self.linear(lstm_out.view(len(input_seq), -1))[-1]
        return out

# In the case of multiclass classification use loss_function = nn.CrossEntropyLoss()


model = LSTMNetwork(hidden_layer_size=10, drop_prob=0.5)
model = model.float()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function_1 = nn.CrossEntropyLoss()
loss_function_2 = nn.MSELoss()

# Train the model
epochs = 100
single_loss = None

for i in range(epochs):
    for seq, labels in Train_Data_Loader:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size), torch.zeros(1, 1, model.hidden_layer_size))
        y_pred = model(seq.float())
        # y_pred = torch.round(y_pred)
        # [[ 4.89349812e-01  4.97627139e-01 -1.08913973e-05] ...
        labels = labels.reshape(-1)
        labels = labels.float()
        # single_loss = loss_function(y_pred, labels)
        single_loss = loss_function_1(y_pred, labels)
        single_loss.backward()
        optimizer.step()
    if i % 10 == 1:
        print(f'epoch: {i} loss: {single_loss.item()}')

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
