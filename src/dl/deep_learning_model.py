import torch

import dl.run_dl as dl
from dl import CNN1D_Classifier as CNN1D
from learning_model_class import LearningModel
from utils import format_data_for_nn as ft


class DeepLearningClassifier(LearningModel):

    def __init__(self, model='model_save/cnn_state_dict.pt', window_size=40):
        self.window_size = window_size

        self.dl_model = CNN1D.CNN1D(self.window_size, "device", output_size=16)
        self.dl_model.eval()
        if model:
            self.dl_model.load_state_dict(torch.load(model))

    def predict_data(self, data) -> any:
        """Ensure that data passed to this function is of the format as returned by read_data function
        This function returns an integer representing the class of the gesture.

        :return: Tupel with predicted class from 0-15 and a confidence value"""

        data = ft.format_individual_data(data)
        data = torch.from_numpy(data)
        pred = self.dl_model.forward(data.view(1, self.window_size, 63).float())

        return torch.argmax(pred).item(), torch.max(pred).item()  # pred_class, confid

    def train_model(self):
        """Assumption - Data is present in HandDataset"""
        run = dl.DL_run(path_to_data="../HandDataset", folder_name="Josh", window_size=self.window_size)

        run.setupDL(CNN1D)
        run.trainDL(CNN1D, lr=0.002, epochs=800)
        run.evalDL(CNN1D)
