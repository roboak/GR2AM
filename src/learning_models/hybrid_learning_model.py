from learning_models.learning_model_class import LearningModel
from src.learning_models.machine_learning_model.machine_learning_model import MachineLearningClassifier
from src.learning_models.neural_network_model.deep_learning_model import DeepLearningClassifier


class HybridLearningClassifier(LearningModel):

    def __init__(self, window_size=40, model_path="saved_models/"):
        self.window_size = window_size
        self.ml = MachineLearningClassifier(already_trained_classifier=model_path + "trained_model.joblib",
                                            window_size=self.window_size)
        self.dl = DeepLearningClassifier(window_size=self.window_size, output_size=18,
                                         model=model_path + 'state_dict.pt')

    def predict_data(self, data):
        """
        :param data: The 30-frame window captured from the live stream
        :return: The prediction results which is computed as depending on the confidence of the models
        """

        result_dl, acc_dl = self.dl.predict_data(data)
        result_ml, acc_ml = self.ml.predict_data(data)

        if acc_dl > 0.82:
            return result_dl
        elif acc_ml > 0.17:
            return result_ml
        else:
            return 15

    def train_model(self):
        pass
