import logging

from dl.deep_learning_model import DeepLearningClassifier
from learning_model_class import LearningModel
from machine_learning_working.machine_learning_model import MachineLearningClassifier


class HybridLearningClassifier(LearningModel):

    def __init__(self):
        self.ml = MachineLearningClassifier(extracted_features_path="extracted_features.joblib")
        # FIXME need to changes this!!
        #self.ml = MachineLearningClassifier(already_trained_classifier="trained_model.joblib")
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
