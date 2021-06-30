import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import load, dump
from src.machine_learning_working.statistical_feature_extraction import FeatureExtraction
from src.utils.read_data import read_data


class MachineLearningClassifier:
    def __init__(self, training_data_path="", training_data_folder="", extracted_features_path="",
                 features_already_extracted: bool = False, already_trained_classifier=None):
        self.training_data_path = training_data_path
        self.training_data_folder = training_data_folder
        self.extracted_features_path = extracted_features_path
        self.feature_extraction = FeatureExtraction()
        self.classifier = already_trained_classifier
        self.features = 0
        self.__fit_classifier(features_already_extracted)

    def __get_features(self):
        """
        This is the function to extract the features for the first time.
        :return:
        """
        training_files, _ = read_data(self.training_data_path, self.training_data_folder)
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
        self.classifier = RandomForestClassifier(random_state=0, n_estimators=40)
        self.features = training_features
        column_number = training_features.shape[1] - 1
        training_data = training_features[:, :column_number]
        training_labels = training_features[:, column_number]
        self.classifier.fit(training_data, training_labels)

    def __fit_classifier(self, features_already_extracted):
        """
        :param features_already_extracted: A boolean to either call and load already available features or to extract
        the features from the start.
        :return:
        """
        if self.classifier is None:
            if features_already_extracted:
                self.__get_features_loaded()
            else:
                self.__get_features()

    def predict_data(self, data):
        """
        :param data: A numpy Array of the data to predict
        :return: The results of the classification
        """
        features = self.feature_extraction.get_features_prediction(data)
        features = np.reshape(features, (1, features.shape[0]))
        prediction_result = self.classifier.predict(features)
        return prediction_result

    def save_features(self):
        if self.features != 0:
            dump(self.features, "extracted_features.joblib")

    def save_model(self):
        dump(self.classifier, "trained_model.joblib")
