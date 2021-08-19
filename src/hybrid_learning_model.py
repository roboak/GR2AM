import logging
import time

from dl.deep_learning_model import DeepLearningClassifier
from learning_model_class import LearningModel
from machine_learning_working.machine_learning_model import MachineLearningClassifier


class HybridLearningClassifier(LearningModel):

    def __init__(self, window_size=40):
        self.window_size = window_size

        self.ml = MachineLearningClassifier(already_trained_classifier="trained_model.joblib", window_size=self.window_size)
        self.dl = DeepLearningClassifier(window_size=self.window_size, output_size=18)

    def predict_data(self, data):

        result_dl, acc_dl = self.dl.predict_data(data)
        result_ml, acc_ml = self.ml.predict_data(data)

        fancy_str = "RESULT--> DL: " + str(result_dl) + ' Confi:' + str(acc_dl)[:6] + " ML: " + str(result_ml) + ' Confi:' + str(acc_ml)[:6]

        if acc_dl > 0.82:
            return result_dl
        elif acc_ml > 0.17:
            return result_ml
        else:
            return 15

        # return fancy_str

        # dl > 82%  --> result
        # else
        # ml > 17% --> result
        # else
        # class 15 --> result

    def train_model(self):
        pass
