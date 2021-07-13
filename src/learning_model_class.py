from abc import ABC, abstractmethod


class LearningModel(ABC):

    @abstractmethod
    def predict_data(self, data):
        pass

    @abstractmethod
    def train_model(self):  # path to the parent folder where the data is present
        pass
