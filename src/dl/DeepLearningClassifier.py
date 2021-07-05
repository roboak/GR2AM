import run_dl as dl
from learning_model_class import LearningModel
from dl import CNN1D_Classifier as CNN1D
import torch
from utils import format_data_for_nn as ft
from utils import read_data as rd

class deep_learning_model(LearningModel):

    """Ensure that data passed to this function is of the format as returned by read_data function
     This function returns an integer representing the class of the gesture."""
    def predict_data(self, data) -> int:
        dl_model = CNN1D.CNN1D(80, "cpu", output_size=15)
        dl_model.eval()
        dl_model.load_state_dict(torch.load('./../model_save/cnn_state_dict.pt'))
        data = ft.format_individual_data(data)
        data = torch.from_numpy(data)
        pred = dl_model.forward(data.view(1, 80, 63).float())
        # result = (labels.item(), torch.argmax(pred).item() + 1, torch.max(pred).item())
        return torch.argmax(pred).item() + 1


    """Assumption - Data is present in HandDataset"""
    def train_model(self):
        run = dl.DL_run(path_to_data="./../../HandDataset", folder_name="Josh")
        print("Training the CNN1D model")
        run.setupDL(CNN1D)
        run.trainDL(CNN1D, lr=0.002, epochs=800)
        run.evalDL(CNN1D)

if (__name__ == '__main__'):
    dl_model = deep_learning_model()
    # dl_model.train_model()
    test_data = rd.read_data("./../../HandDataset/TrainingData", "Josh",predef_size = 80 )
    print("prediction: ", dl_model.predict_data(test_data[0][1].data))
    print("label: ", test_data[0][1].label)
