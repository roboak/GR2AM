from abc import ABC, abstractmethod


class LearningModel(ABC):

    @abstractmethod
    def predict_data(self, data):
        pass

    @abstractmethod
    def save_features(self):
        pass

    @abstractmethod
    def save_model(self):
        pass
