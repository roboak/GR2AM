import os
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

from utils import format_data_for_nn, read_data
from dl import CNN1D_Classifier as CNN1D
from dl import CNN_GRU as CNN_GRU
from dl import GRU_Classifier as GRU

class DL_run:
    def __init__(self):
        self.val_loader = None
        self.train_loader = None
        self.batch_size = 4
        self.test_batch_size = 1
        self.device = format_data_for_nn.get_device()
        # device = "cpu"

    def setupDL(self, obj):
        dataset, seq_len = read_data.read_data()
        num_classes, data_dict = format_data_for_nn.format_data(dataset=dataset)
        #X_train, X_test, X_val, y_train, y_test, y_val = format_data_for_nn.split_training_test_valid(
        #    data_dict=data_dict,
        #    num_labels=num_classes)
        #self.train_loader, self.val_loader, self.test_loader = format_data_for_nn.get_mini_batches(X_train, X_test,
        #                                                                                           X_val, y_train,
        #                                                                                           y_test,
        #                                                                                           y_val,
        #                                                                                           self.batch_size,
        #                                                                                           test_batch_size=self.test_batch_size)

        #self.model = obj.CNN_LSTM(seq_len, self.device, output_size=num_classes).to(self.device)
        self.model = obj.CNN_LSTM(86, self.device, output_size=3).to(self.device)

        data_dict["labels"] = data_dict["labels"] - 1
        test_dataset = TensorDataset(torch.from_numpy(data_dict["data"]), torch.from_numpy(data_dict["labels"]))
        self.test_loader = DataLoader(test_dataset, shuffle=True, batch_size=1, drop_last=True)

    def trainDL(self, obj, lr=0.002, epochs=100):
        self.nn_train = obj.train_neural_network(model=self.model, device=self.device, batch_size=self.batch_size,
                                                 lr=lr, epochs=epochs, train_loader=self.train_loader,
                                                 test_loader=self.test_loader,
                                                 val_loader=self.val_loader)
        #self.nn_train.train_model()

    def evalDL(self, obj):
        self.model.load_state_dict(torch.load('../model_save/cnn_state_dict.pt'))
        #self.model.load_state_dict(torch.load('../model_save/cnn_gru_state_dict.pt'))
        #self.model.load_state_dict(torch.load('../model_save/gru_state_dict.pt'))
        self.nn_train.evaluate_model(self.test_batch_size)


if __name__ == '__main__':
    run = DL_run()

    if True:
        print("RUNNING CNN1D")
        run.setupDL(CNN1D)
        run.trainDL(CNN1D, lr=0.005, epochs=70)
        run.evalDL(CNN1D)

    if False:
        print("RUNNING CNN_GRU")
        run.setupDL(CNN_GRU)
        run.trainDL(CNN_GRU, lr=0.002, epochs=100)
        run.evalDL()

    if False:
        print("RUNNING GRU")
        run.setupDL(GRU)
        run.trainDL(GRU, lr=0.005, epochs=100)
        run.evalDL()
