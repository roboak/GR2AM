import numpy as np
from joblib import load
from sklearn.ensemble import RandomForestClassifier

from src.machine_learning_working.statistical_feature_extraction import FeatureExtraction
from src.utils.read_data import read_data


class Classifier:
    def __init__(self, training_data_path="", training_data_folder=""):
        self.training_data_path = training_data_path
        self.training_data_folder = training_data_folder
        self.feature_extraction = FeatureExtraction()
        self.classifier = RandomForestClassifier(random_state=0, n_estimators=40)

    def get_classifier(self):
        training_files, _ = read_data(self.training_data_path, self.training_data_folder)
        training_features = np.asarray([self.feature_extraction.get_features_training(file) for file in training_files])
        column_number = training_features.shape[1] - 1
        training_data = training_features[:, :column_number]
        training_labels = training_features[:, column_number]
        # labels = set(training_labels.tolist())
        # labels_display = [f"gesture_{int(i)}" for i in list(labels)]
        self.classifier.fit(training_data, training_labels)
        return self.classifier

    def get_classifier_loaded(self, file: str):
        training_features = load(file)
        column_number = training_features.shape[1] - 1
        training_data = training_features[:, :column_number]
        training_labels = training_features[:, column_number]
        self.classifier.fit(training_data, training_labels)
        return self.classifier
