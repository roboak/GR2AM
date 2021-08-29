import joblib
import numpy as np
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from src.learning_models.learning_model_class import LearningModel
from src.learning_models.machine_learning_model.statistical_feature_extraction import FeatureExtraction
from src.utils.gesture_data_related.read_data import read_data


class MachineLearningClassifier(LearningModel):
    def __init__(self, training_data_path="", training_data_folder="", extracted_features_path="",
                 already_trained_classifier=None, window_size=40):
        self.training_data_path = training_data_path
        self.training_data_folder = training_data_folder
        self.extracted_features_path = extracted_features_path
        self.feature_extraction = FeatureExtraction()
        self.classifier = joblib.load(already_trained_classifier) if already_trained_classifier is not None else None
        self.features = 0
        self.window_size = window_size
        self.__fit_classifier()

    def __get_features(self):
        """
        This is the function to extract the features for the first time.
        :return:
        """
        training_files, _ = read_data(self.training_data_path, self.training_data_folder, self.window_size)
        training_features = np.asarray([self.feature_extraction.get_features_training(file) for file in training_files])
        self.__train_classifier(training_features)

    def __get_features_loaded(self):
        """
        This is the function to load already extracted features
        :return:
        """
        training_features = load(self.extracted_features_path)
        self.__train_classifier(training_features)

    def __train_classifier(self, training_features: np.array):
        """
        This is the function to train the classifier
        :param training_features:
        :return:
        """
        self.classifier = RandomForestClassifier(random_state=0, n_estimators=400)
        # n_estimators: The number of trees in the forest.
        self.features = training_features
        column_number = training_features.shape[1] - 1
        training_data = training_features[:, :column_number]
        training_labels = training_features[:, column_number]
        self.classifier.fit(training_data, training_labels)

    def __fit_classifier(self):
        """
        :param
        :return:
        """
        if self.classifier is None:
            if self.extracted_features_path:
                self.__get_features_loaded()
            else:
                self.__get_features()

    def predict_data(self, data: np.array):
        """
        :param data: A numpy Array of the data to predict
        :return: The results of the classification
        """
        features = self.feature_extraction.get_features_prediction(data)
        features = np.reshape(features, (1, features.shape[0]))
        prediction_result = self.classifier.predict(features)
        prediction_percentage_array = self.classifier.predict_proba(features)
        print(prediction_percentage_array)
        return int(prediction_result[0]) - 1, np.amax(prediction_percentage_array)

    def save_features(self):
        if self.features != 0:
            dump(self.features, "./saved_models/extracted_features.joblib")

    def save_model(self, save_path="./saved_models/trained_model.joblib"):
        dump(self.classifier, save_path)

    def train_model(self):
        pass
