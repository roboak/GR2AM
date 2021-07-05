import os
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from utils import format_data_for_nn
from dl import CNN1D_Classifier as CNN1D
from dl import CNN_GRU as CNN_GRU
from dl import GRU_Classifier as GRU
from joblib import dump, load

class DL_run:
    def __init__(self, path_to_data, folder_name):
        self.val_loader = None
        self.train_loader = None
        self.batch_size = 32
        self.test_batch_size = 1
        self.val_batch_size = 32
        self.device = format_data_for_nn.get_device()
        self.path_to_data = path_to_data
        self.folder_name = folder_name
        # device = "cpu"

    def setupDL(self, obj):
        # dataset, seq_len = read_data.read_data()
        # num_classes, data_dict = format_data_for_nn.format_data(dataset=dataset)
        # X_train, X_test, X_val, y_train, y_test, y_val = format_data_for_nn.split_training_test_valid(
        #    data_dict=data_dict,
        #    num_labels=num_classes)
        # self.train_loader, self.val_loader, self.test_loader = format_data_for_nn.get_mini_batches(X_train, X_test,
        #                                                                                           X_val, y_train,
        #                                                                                           y_test,
        #                                                                                           y_val,
        #                                                                                           self.batch_size,
        #                                                                                           test_batch_size=self.test_batch_size,
        #                                                                                            val_batch_size= self.val_batch_size)

        self.train_loader, self.val_loader, self.test_loader, seq_len, num_classes = format_data_for_nn.get_data_for_training(batch_size= self.batch_size,
                                                                                                                              val_batch_size=self.val_batch_size,
                                                                                                                              test_batch_size=self.test_batch_size,
                                                                                                                              path_to_data= self.path_to_data ,
                                                                                                                              folder_name = self.folder_name
                                                                                                                              )
        self.model = obj.CNN1D(seq_len, self.device, output_size=num_classes).to(self.device)

    def trainDL(self, obj, lr=0.002, epochs=100):
        self.nn_train = obj.train_neural_network(model=self.model, device=self.device,
                                                 lr=lr, epochs=epochs, train_loader=self.train_loader,
                                                 test_loader=self.test_loader,
                                                 val_loader=self.val_loader)
        self.nn_train.train_model()

    def evalDL(self, obj):
        self.model.load_state_dict(torch.load('../model_save/cnn_state_dict.pt'))
        # self.model.load_state_dict(torch.load('../cnn_state_dict_abdul_95.pt'))
        # self.model.load_state_dict(torch.load('../model_save/gru_state_dict.pt'))
        self.nn_train.evaluate_model(self.test_batch_size)


if __name__ == '__main__':
    run = DL_run()

    if True:
        print("RUNNING CNN1D")
        run.setupDL(CNN1D, path_to_data, folder_name)
        run.trainDL(CNN1D, lr=0.002, epochs=800)
        run.evalDL(CNN1D)


    if False:
        print("RUNNING CNN_GRU")
        run.setupDL(CNN_GRU, path_to_data, folder_name)
        run.trainDL(CNN_GRU, lr=0.002, epochs=100)
        run.evalDL(CNN_GRU)

    if False:
        print("RUNNING GRU")
        run.setupDL(GRU)
        run.trainDL(GRU, lr=0.005, epochs=100)
        run.evalDL()
