import torch

import src.dl.run_dl as dl
from src.dl import CNN1D_Classifier as CNN1D
from src.learning_model_class import LearningModel
from src.utils import format_data_for_nn as ft, read_data as rd


class DeepLearningClassifier(LearningModel):

    def __init__(self, window_size, model='model_save/cnn_state_dict.pt', output_size=18):
        self.window_size = window_size
        self.output_size = output_size

        self.dl_model = CNN1D.CNN1D(self.window_size, "device", output_size=self.output_size)
        self.dl_model.eval()
        if model:
            self.dl_model.load_state_dict(torch.load(model))

    def predict_data(self, data) -> any:
        """Ensure that data passed to this function is of the format as returned by read_data function
        This function returns an integer representing the class of the gesture.

        :return: Tupel with predicted class from 0-15 and a confidence value"""

        #dl_model = CNN1D.CNN1D(self.window_size, "device", output_size=16)
        #dl_model.eval()
        #dl_model.load_state_dict(torch.load('model_save/cnn_state_dict.pt'))
        
        data = ft.format_individual_data(data)
        data = torch.from_numpy(data)
        pred = self.dl_model.forward(data.view(1, self.window_size, 63).float())

        return torch.argmax(pred).item(), torch.max(pred).item()  # pred_class, confid

    def train_model(self):
        """Assumption - Data is present in HandDataset"""
        run = dl.DL_run(path_to_data="../HandDataset", folder_name="Josh2_less", window_size=self.window_size, )
        run.setupDL(CNN1D, output_size=self.output_size)
        run.trainDL(CNN1D, lr=0.002, epochs=800)
        run.evalDL(CNN1D)


if __name__ == '__main__':
    dl_model = DeepLearningClassifier()
    # dl_model.train_model()
    test_data = rd.read_data("./../../HandDataset/TrainingData", "Josh", predef_size=dl_model.window_size)
    print("prediction: ", dl_model.predict_data(test_data[0][1].data))
    print("label: ", test_data[0][1].label)
