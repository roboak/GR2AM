import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix
import read_data



class LSTMNetwork(nn.Module):
    def __init__(self, device, input_size=63, hidden_dim=128, output_size=3, n_layer=64, drop_prob=0.1):
        super(LSTMNetwork, self).__init__()
        self.device = device
        # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
        self.hidden_dim = hidden_dim
        self.n_layers = n_layer

        # #Converting (batch_size,seq_len,63) ->
        # self.embedding_dim = 64
        # self.embedding_layer = nn.Embedding(
        #     num_embeddings=output_size,
        #     embedding_dim=self.embedding_dim,
        #     padding_idx=0
        # )
        # If your input data is of shape (batch_size, seq_len, features)
        # then you need batch_first=True and your LSTM will give output of shape (batch_size, seq_len, hidden_size).
        self.lstm = nn.LSTM(input_size, self.hidden_dim, self.n_layers,
                            batch_first=True)  # input_dim, hidden_dim, n_layers, batch_first=True
        self.dropout = nn.Dropout(drop_prob)
        self.linear = nn.Linear(hidden_dim, output_size)
        # self.linear_2 = nn.Linear(63, output_size)
        self.activation = nn.Softmax()

    def forward(self, x, hidden):
        batch_size = x.size(0)

        #converting (batch_size, seq_len, 63) -> (batch_size, seq_len, 128)
        # embedding_out = self.embedding_layer(x)
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
        weight = next(self.parameters()).data
        # hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
        #           weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))

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
        self.criterion = nn.CrossEntropyLoss()
        self.model = model
        self.optimiser = torch.optim.Adam(model.parameters(), lr=self.lr)

    def train_model(self):
        counter = 0
        print_every = 10
        clip = 5
        valid_loss_min = np.Inf
        # model.train() tells your model that you are training the model. So effectively layers like dropout, batchnorm etc.
        # which behave different on the train and test procedures know what is going on and hence can behave accordingly.
        self.model.train()
        for i in range(self.epochs):
            h = self.model.init_hidden(self.batch_size)
            train_losses =[]
            for inputs, labels in self.train_loader:
                counter += 1
                h = tuple([e.data for e in h])
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                #self.model.zero_grad() #manually setting all the gradients to zero
                self.optimiser.zero_grad()
                output, h = self.model.forward(inputs.float(), h)
                #         print (output.shape)
                #         print (h[0].shape)
                # loss = criterion(output.squeeze(), labels.float())
                loss = self.criterion(output, labels.long())#float())
                train_losses.append(loss)
                loss.backward()
                #nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                self.optimiser.step()

                if counter % print_every == 0:
                    val_h = self.model.init_hidden(self.batch_size)
                    val_losses = []
                    self.model.eval()
                    for inp, lab in self.val_loader:
                        val_h = tuple([each.data for each in val_h])
                        inp, lab = inp.to(self.device), lab.to(self.device)
                        out, val_h = self.model.forward(inp.float(), val_h)  # model(inp, val_h)
                        # val_loss = criterion(out.squeeze(), lab.float())
                        val_loss = self.criterion(out, lab.long())#float())
                        val_losses.append(val_loss.item())

                    self.model.train()
                    print("Epoch: {}/{}...".format(i + 1, self.epochs),
                          "Step: {}...".format(counter),
                          "Loss: {:.6f}...".format(loss.item()),
                          "Val Loss: {:.6f}".format(np.mean(val_losses)))
                    if np.mean(val_losses) <= valid_loss_min:
                        torch.save(self.model.state_dict(), 'model_save/state_dict.pt')
                        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,
                                                                                                        np.mean(
                                                                                                            val_losses)))
                        valid_loss_min = np.mean(val_losses)
            # plt.plot(train_losses)
            # plt.ylabel('losses')
            # plt.show()
        # return self.model

    def evaluate_model(self):
        test_losses = []
        num_correct = 0
        h = model.init_hidden(batch_size)
        model.eval()
        confusion_matrix = None
        for inputs, labels in test_loader:
            h = tuple([each.data for each in h])
            inputs, labels = inputs.to(device), labels.to(device)
            output, h = model.forward(inputs.float(), h)
            test_loss = self.criterion(output, labels.long())#float())
            test_losses.append(test_loss.item())
            #pred = torch.round(output)  # rounds the output to 0/1
            print(output)
            print(labels)
            # pred = torch.where(output>0.75, 1.0, 0.0)
            # correct_tensor = pred - labels  # pred.eq(labels.float().view_as(pred))
            # print(correct_tensor)
            # if not (-1 in correct_tensor):
            #     num_correct += 1
            #
            # confusion_matrix = multilabel_confusion_matrix(labels.detach().numpy(), pred.detach().numpy())
            # print("confusion_matrix:", confusion_matrix)

        # print("Test loss: {:.3f}".format(np.mean(test_losses)))
        # test_acc = num_correct / len(test_loader.dataset)
        # print("Test accuracy: {:.3f}%".format(test_acc * 100))

        # plt.plot(test_losses)
        # plt.ylabel('losses')
        # plt.show()

#
# batch_size = 4
# dataset = read_data.read_data()
# num_classes, data_dict = format_data(dataset=dataset)
# X_train, X_test, X_val, y_train, y_test, y_val = split_training_test_valid(data_dict=data_dict, num_labels=num_classes)
# train_loader, val_loader, test_loader = get_mini_batches(X_train, X_test, X_val, y_train, y_test, y_val, batch_size)
# device = get_device()
# # device = "cpu"
# model = LSTMNetwork(device)
# model.to(device)
# # print(model)
# nn_train = train_neural_network(model=model, device=device, batch_size=batch_size,
#                                 lr=0.001, epochs=10, train_loader=train_loader, test_loader=test_loader,
#                                 val_loader=val_loader)
# nn_train.train_model()
#
# # model.load_state_dict(torch.load('state_dict.pt'))
# nn_train.evaluate_model()
