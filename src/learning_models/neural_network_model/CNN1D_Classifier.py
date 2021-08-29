import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.nn import BatchNorm1d, Conv1d, Dropout, Flatten, Linear, Sequential, Softmax, Tanh


# from torch.utils.tensorboard import SummaryWriter
# torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
# padding_mode='zeros')

class CNN1D(nn.Module):
    def __init__(self, seq_len, device, output_size, n_layers=2, hidden_dim=128, ):
        super(CNN1D, self).__init__()
        # self.device = device
        self.output_size = output_size
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.n_cnn_filter_1 = 128
        self.n_cnn_filter_2 = 64
        self.cnn_kernel_size_1 = 1
        self.cnn_kernel_size_2 = 1
        self.cnn_stride_1 = 1
        self.cnn_stride_2 = 1

        self.cnn_layers = Sequential(
            # out_channel = number of filters in the CNN
            Conv1d(in_channels=self.seq_len, out_channels=self.n_cnn_filter_1,
                   kernel_size=self.cnn_kernel_size_1, stride=self.cnn_stride_1),
            BatchNorm1d(self.n_cnn_filter_1),
            # ReLU(),
            Tanh(),
            Dropout(0.15),
            # MaxPool1d(2),
            # len_output_features_per_frame = int((input_features_per_frame - kernel_size)/stride) + 1
            Conv1d(in_channels=self.n_cnn_filter_1, out_channels=self.n_cnn_filter_2,
                   kernel_size=self.cnn_kernel_size_2, stride=self.cnn_stride_2),
            BatchNorm1d(self.n_cnn_filter_2),
            # ReLU(),
            Tanh(),
            Dropout(0.10),
            # MaxPool1d(2),
            # len_output_features_per_frame_after_pooling = [int((input_features_per_frame - kernel_size)/stride) + 1]//2

        )
        self.flatten = Flatten()
        cnn_output1 = (63 - self.cnn_kernel_size_1) // self.cnn_stride_1 + 1
        max_pool1 = cnn_output1 // 1
        cnn_output2 = (max_pool1 - self.cnn_kernel_size_2) // self.cnn_stride_2 + 1
        max_pool2 = cnn_output2 // 1
        self.dense_layers = Sequential(
            Linear(self.n_cnn_filter_2 * max_pool2, self.output_size),
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
    model = CNN1D(seq_len, device).to(device)
    out, hidden = model.forward(inp, model.init_hidden(batch_size))
    print(out.shape)
    # out = out.view(20, 64, 3)
    print(out.shape)
    # out = out[:, -1:, :].squeeze()
    print(out.shape)


class train_neural_network:
    def __init__(self, model, device, lr, epochs, train_loader, test_loader, val_loader):
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.criterion = nn.CrossEntropyLoss()
        self.model = model
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train_model(self, model_save_path='saved_models/cnn_state_dict.pt'):
        counter = 0
        print_every = 10
        clip = 15
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
                    loss.backward()
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
                            torch.save(self.model.state_dict(), model_save_path)
                            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                                valid_loss_min,
                                np.mean(
                                    val_losses)))
                            valid_loss_min = np.mean(val_losses)

    def evaluate_model(self, test_batch_size, img_path="saved_figure.png"):
        test_losses = []
        num_correct = 0
        self.model.eval()

        pred_list = []
        label_list = []

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
            num_correct += torch.sum((pred == labels))

            label_list += labels.tolist()
            pred_list += pred.tolist()

        print("label_list_size: {}".format(label_list))
        print("pred_list_size: {}".format(pred_list))
        #
        confusi = confusion_matrix(label_list, pred_list, labels=[x for x in range(self.model.output_size)])
        # print("confusion_matrix:\n", confusi)
        display_1 = ConfusionMatrixDisplay(confusion_matrix=confusi,
                                           display_labels=["gesture" + str(x) for x in range(1, self.model.output_size+1)]).plot()

        print("Test loss: {:.3f}".format(np.mean(test_losses)))
        test_acc = num_correct / (test_batch_size * num_test_mini_batches)
        print("Test accuracy: {:.3f}%".format(test_acc * 100))
        fig = plt.gcf()
        fig.savefig(img_path)
        plt.show()

