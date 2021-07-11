import logging
import sys

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix

from src.dl import CNN1D_Classifier as CNN1D
from src.dl.deep_learning_model import DeepLearningClassifier
from src.learning_model_class import LearningModel
from src.machine_learning_working.get_classifier import Classifier
from src.machine_learning_working.machine_learning_model import MachineLearningClassifier
from src.machine_learning_working.predict import Predict
from src.utils import format_data_for_nn as ft, read_data as rd


class HybridLearningClassifier(LearningModel):

    def __init__(self):
        self.ml = MachineLearningClassifier(extracted_features_path="extracted_features.joblib")
        self.dl = DeepLearningClassifier()

    def predict_data(self, data):
        result_dl, acc_dl = self.dl.predict_data(data)
        result_ml = self.ml.predict_data(data)

        # TODO could we do a combined score instead?
        print(acc_dl)
        logging.debug("DL: " + str(result_dl) + " ML: " + str(result_ml) + ' Confi:' + str(acc_dl))
        if acc_dl < 0.85:
            print("ML" + str(result_ml))
            return result_ml
        else:
            print("DL: " + str(result_dl))
            return int(result_dl)

    def train_model(self):
        pass
