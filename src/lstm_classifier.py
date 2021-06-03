import numpy as np
import read_data
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn

def format_data(dataset): #read data in the format of [total_data_size, sequence length, feature_size, feature_dim]
    seq_len = dataset[0].data.shape[0]
    num_features = dataset[0].data.shape[1]
    feature_dim = dataset[0].data.shape[2]
    data_array = np.zeros((len(dataset), seq_len, num_features, feature_dim))
    labels = np.zeros(len(dataset))
    data_dict = {}
    for idx, data in enumerate(dataset):
        data_array[idx] = data.data
        labels[idx] = data.label
    # Return all data stacked together
    # Shape of data changes from [batch_size,seq_len,input_dim,feature_dims] -> [batch_size,seq_len,input_dim*feature_dims]
    data_dict["data"] = data_array.reshape(len(data_array), seq_len, num_features*feature_dim).astype(np.float)
    data_dict["labels"] = labels.astype(int)
    num_classes = len(np.unique(data_dict["labels"]))
    return num_classes, data_dict

def hot_encoding(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y-1]

def to_categorical(labels, num_classes):
    new_labels = np.zeros((len(labels), num_classes))
    for idx, label in enumerate(labels):
        new_labels[idx] = hot_encoding(label, num_classes)
    return new_labels

def split_training_test_valid(data_dict, num_labels):
    data_dict["labels"] = to_categorical(data_dict["labels"], num_labels)
    X_train, X_test, y_train, y_test = train_test_split(data_dict["data"], data_dict["labels"], test_size=0.3,random_state=42)
    split_frac = 0.5
    split_id = int(split_frac * len(X_test))
    X_val, X_test = X_test[:split_id], X_test[split_id:]
    y_val, y_test = y_test[:split_id], y_test[split_id:]
    return X_train, X_test, X_val, y_train, y_test, y_val


def get_mini_batches(X_train, X_test, X_val, y_train, y_test, y_val, batch_size):
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)
    return train_loader, val_loader, test_loader

def get_device():
    is_cuda = torch.cuda.is_available()
    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")
    return device

class LSTMNetwork(nn.Module):
    def __init__(self, device, input_size=63, hidden_dim=100, output_size=3, n_layer=1, drop_prob=0.2):
        super(LSTMNetwork, self).__init__()
        self.device = device
        # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
        self.hidden_dim = hidden_dim
        self.n_layers = n_layer
        # If your input data is of shape (batch_size, seq_len, features)
        # then you need batch_first=True and your LSTM will give output of shape (batch_size, seq_len, hidden_size).
        self.lstm = nn.LSTM(input_size, self.hidden_dim, self.n_layers,
                            batch_first=True)  # input_dim, hidden_dim, n_layers, batch_first=True
        self.dropout = nn.Dropout(drop_prob)
        self.linear = nn.Linear(hidden_dim, output_size)
        # self.linear_2 = nn.Linear(63, output_size)
        self.activation = nn.Softmax()
        # self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size), torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, x, hidden):
        batch_size = x.size(0)
        # x = x.long()
        lstm_out, hidden = self.lstm(x, hidden)
        # https://stackoverflow.com/questions/54749665/stacking-up-of-lstm-outputs-in-pytorch
        #   print ("lstm_out_ before reshaping", lstm_out.shape)
        # Converting [batch_size x seq_len _ X hiddednn_dim] -> [batch_size * seq_len  X hiddednn_dim]
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        #   print ("lstm_out_ after reshaping", lstm_out.shape)
        out = self.dropout(lstm_out)
        out = self.linear(out)
        out = self.activation(out)
        #  print ("Dense layer output  dimension: ",out.shape)
        out = out.view(batch_size, x.size(1), -1)
        # print ("Dense layer reshaped1 output  dimension: ",out.shape)
        out = out[:, -1, :]
        # print ("Dense layer reshaped2 output  dimension: ",out.shape)
        return out, hidden

    def init_hidden(self, batch_size):
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device),
                  torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device))
        return hidden

class train_neural_network:
    def __init__(self, model, device, batch_size, lr, epochs, train_loader, test_loader, val_loader):
        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.criterion = nn.BCELoss()
        self.model = model
        self.optimiser = torch.optim.Adam(model.parameters(), lr=self.lr)


    def train_model(self):
        counter = 0
        print_every = 2
        clip = 5
        valid_loss_min = np.Inf
        # model.train() tells your model that you are training the model. So effectively layers like dropout, batchnorm etc.
        # which behave different on the train and test procedures know what is going on and hence can behave accordingly.
        self.model.train()
        for i in range(self.epochs):
            h = self.model.init_hidden(self.batch_size)

            for inputs, labels in self.train_loader:
                counter += 1
                h = tuple([e.data for e in h])
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.model.zero_grad()
                output, h = self.model.forward(inputs.type(torch.cuda.FloatTensor), h)
                #         print (output.shape)
                #         print (h[0].shape)
                # loss = criterion(output.squeeze(), labels.float())
                loss = self.criterion(output, labels.float())
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                self.optimiser.step()

                if counter % print_every == 0:
                    val_h = self.model.init_hidden(self.batch_size)
                    val_losses = []
                    self.model.eval()
                    for inp, lab in self.val_loader:
                        val_h = tuple([each.data for each in val_h])
                        inp, lab = inp.to(self.device), lab.to(self.device)
                        out, val_h = self.model.forward(inputs.type(torch.cuda.FloatTensor), val_h)  # model(inp, val_h)
                        # val_loss = criterion(out.squeeze(), lab.float())
                        val_loss = self.criterion(out, lab.float())
                        val_losses.append(val_loss.item())

                    self.model.train()
                    print("Epoch: {}/{}...".format(i + 1, self.epochs),
                          "Step: {}...".format(counter),
                          "Loss: {:.6f}...".format(loss.item()),
                          "Val Loss: {:.6f}".format(np.mean(val_losses)))
                    if np.mean(val_losses) <= valid_loss_min:
                        torch.save(self.model.state_dict(), './state_dict.pt')
                        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,
                                                                                                        np.mean(
                                                                                                            val_losses)))
                        valid_loss_min = np.mean(val_losses)
        # return self.model

    def evaluate_model(self):
        test_losses = []
        num_correct = 0
        h = model.init_hidden(batch_size)
        model.eval()
        for inputs, labels in test_loader:
            h = tuple([each.data for each in h])
            inputs, labels = inputs.to(device), labels.to(device)
            output, h = model.forward(inputs.type(torch.cuda.FloatTensor), h)
            test_loss = self.criterion(output, labels.float())
            test_losses.append(test_loss.item())
            pred = torch.round(output)  # rounds the output to 0/1
            correct_tensor = pred - labels  # pred.eq(labels.float().view_as(pred))
            print(correct_tensor)
            if (not (-1 in correct_tensor)):
                num_correct += 1
        print("Test loss: {:.3f}".format(np.mean(test_losses)))
        test_acc = num_correct / len(test_loader.dataset)
        print("Test accuracy: {:.3f}%".format(test_acc * 100))


batch_size = 1
dataset = read_data.read_data()
num_classes, data_dict = format_data(dataset=dataset)
X_train, X_test, X_val, y_train, y_test, y_val = split_training_test_valid(data_dict=data_dict, num_labels=num_classes)
train_loader, val_loader, test_loader = get_mini_batches(X_train, X_test, X_val, y_train, y_test, y_val, batch_size)
device = get_device()
#device = "cpu"
model = LSTMNetwork(device)
model.to(device)
#print(model)
nn_train = train_neural_network(model=model, device=device, batch_size= batch_size,
                                lr=0.005, epochs = 10, train_loader = train_loader, test_loader = test_loader, val_loader = val_loader)
nn_train.train_model()
nn_train.evaluate_model()