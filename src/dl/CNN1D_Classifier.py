import numpy as np
import torch
import torch.nn as nn
from torch.nn import BatchNorm1d, Conv1d, Dropout, Flatten, Linear, MaxPool1d, Sequential, Softmax, Tanh

from utils import format_data_for_nn, read_data


# from torch.utils.tensorboard import SummaryWriter

# torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
class CNN_LSTM(nn.Module):
    def __init__(self, seq_len, device, n_layers=2, hidden_dim=128, output_size=3):
        super(CNN_LSTM, self).__init__()
        self.device = device
        self.output_size = output_size
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.cnn_layers = Sequential(
            # out_channel = number of filters in the CNN
            Conv1d(in_channels=self.seq_len, out_channels=64, kernel_size=1, stride=1),
            BatchNorm1d(64),
            Tanh(),
            Dropout(0.12),
            MaxPool1d(2),

            Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride=1),
            BatchNorm1d(64),
            Tanh(),
            Dropout(0.2),
            MaxPool1d(2),
        )
        self.flatten = Flatten()
        self.dense_layers = Sequential(
            Linear(64 * 15, self.output_size),
            Softmax()
        )

    def forward(self, input):
        cnn_out = self.cnn_layers(input).clone()
        flat_out = self.flatten(cnn_out)
        dense_out = self.dense_layers(flat_out)
        return dense_out


def debug():
    device = "cpu"  # lstm_classifier.get_device()
    seq_len = 50
    batch_size = 20
    inp = torch.randn(batch_size, seq_len, 63).to(device)
    model = CNN_LSTM(seq_len, device).to(device)
    out, hidden = model.forward(inp, model.init_hidden(batch_size))
    print(out.shape)
    # out = out.view(20, 64, 3)
    print(out.shape)
    # out = out[:, -1:, :].squeeze()
    print(out.shape)


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
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train_model(self):
        counter = 0
        print_every = 10
        clip = 5
        valid_loss_min = np.Inf
        # model.train() tells your model that you are training the model. So effectively layers like dropout, batchnorm etc.
        # which behave different on the train and test procedures know what is going on and hence can behave accordingly.
        self.model.train()
        for i in range(self.epochs):
            train_losses = []
            for inputs, labels in self.train_loader:
                with torch.autograd.set_detect_anomaly(True):
                    counter += 1
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    self.model.zero_grad()  # manually setting all the gradients to zero
                    # self.optimiser.zero_grad()
                    output = self.model.forward(inputs.float())
                    loss = self.criterion(output, labels.long())  # float())
                    train_losses.append(loss)
                    loss.backward()  # retain_graph=True)
                    nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                    self.optimiser.step()

                    if counter % print_every == 0:
                        val_losses = []
                        self.model.eval()
                        for inp, lab in self.val_loader:
                            inp, lab = inp.to(self.device), lab.to(self.device)
                            out = self.model.forward(inp.float())  # model(inp, val_h)
                            val_loss = self.criterion(out, lab.long())  # float())
                            val_losses.append(val_loss.item())

                        self.model.train()
                        print("Epoch: {}/{}...".format(i + 1, self.epochs),
                              "Step: {}...".format(counter),
                              "Loss: {:.6f}...".format(loss.item()),
                              "Val Loss: {:.6f}".format(np.mean(val_losses)))
                        if np.mean(val_losses) <= valid_loss_min:
                            torch.save(self.model.state_dict(), '../model_save/cnn_state_dict.pt')
                            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                                valid_loss_min,
                                np.mean(
                                    val_losses)))
                            valid_loss_min = np.mean(val_losses)
            # plt.plot(train_losses)
            # plt.ylabel('losses')
            # plt.show()
        # return self.model

    def evaluate_model(self, test_batch_size):
        test_losses = []
        num_correct = 0
        self.model.eval()
        confusion_matrix = None
        num_test_mini_batches = 0
        for inputs, labels in self.test_loader:
            num_test_mini_batches += 1
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            output = self.model.forward(inputs.float())
            output = output.view(test_batch_size, -1)
            test_loss = self.criterion(output, labels.long())  # float())
            test_losses.append(test_loss.item())
            print("raw out", output)
            pred = torch.argmax(output, dim=1)
            # print("pred1:", pred)
            print("labels:", labels)
            num_correct += torch.sum((pred == labels))

            # confusion_matrix = multilabel_confusion_matrix(labels.detach().numpy(), pred.detach().numpy())
            # print("confusion_matrix:", confusion_matrix)

        print("Test loss: {:.3f}".format(np.mean(test_losses)))
        test_acc = num_correct / (test_batch_size * num_test_mini_batches)
        print("Test accuracy: {:.3f}%".format(test_acc * 100))

        # plt.plot(test_losses)
        # plt.ylabel('losses')
        # plt.show()


# # debug()
# batch_size = 4
# test_batch_size = 1
# device = format_data_for_nn.get_device()
# # device = "cpu"
# dataset, seq_len = read_data.read_data()
# num_classes, data_dict = format_data_for_nn.format_data(dataset=dataset)
# X_train, X_test, X_val, y_train, y_test, y_val = format_data_for_nn.split_training_test_valid(data_dict=data_dict,
#                                                                                               num_labels=num_classes)
# train_loader, val_loader, test_loader = format_data_for_nn.get_mini_batches(X_train, X_test, X_val, y_train, y_test,
#                                                                             y_val, batch_size,
#                                                                             test_batch_size=test_batch_size)
# model = CNN_LSTM(seq_len, device, output_size=num_classes).to(device)
# # print(model)
# nn_train = train_neural_network(model=model, device=device, batch_size=batch_size,
#                                 lr=0.005, epochs=70, train_loader=train_loader, test_loader=test_loader,
#                                 val_loader=val_loader)
# nn_train.train_model()
#
# # model.load_state_dict(torch.load('../model_save/cnn_state_dict.pt'))
# nn_train.evaluate_model(test_batch_size)
