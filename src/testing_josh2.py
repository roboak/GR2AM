from os.path import abspath, dirname
from pathlib import Path
import numpy as np
from joblib import load, dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from utils.read_data import read_data
from machine_learning_working.statistical_feature_extraction import FeatureExtraction


def main():
    window_size = 20
    parent_directory = dirname(dirname(abspath(__file__)))
    parent_directory = Path(parent_directory)
    path = parent_directory / "HandDataset"
    # extracted_features_path = "extracted_features.joblib"
    # training_features = load(extracted_features_path)
    feature_extraction = FeatureExtraction()
    train_data, _ = read_data(path=str(path/"TrainingData"), sub_path="Akash_new", predef_size=window_size)
    features_labels_training = np.asarray([feature_extraction.get_features_training(file) for file in train_data])
    # dump(features_labels_training, "abdul_training.joblib")
    # features_labels_training = load("abdul_training.joblib")
    training_labels = features_labels_training[:, -1]
    training_features = features_labels_training[:, :-1]
    classifier = RandomForestClassifier(random_state=0, n_estimators=40)
    classifier.fit(training_features, training_labels)
    test_data, _ = read_data(path=str(path/"TestingData"), sub_path="Akash_new", predef_size=window_size)
    features_labels = np.asarray([feature_extraction.get_features_training(file) for file in test_data])
    # dump(features_labels, "abdul_testing.joblib")
    # features_labels = load("abdul_testing.joblib")
    labels = features_labels[:, -1]
    features = features_labels[:, :-1]
    predicted_labels = classifier.predict(features)
    confusion_matrix_1 = confusion_matrix(labels, predicted_labels)
    print(confusion_matrix_1)
    print(f"Accuracy {accuracy_score(labels, predicted_labels)} ")
    display_1 = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_1)
    display_1.plot()
    plt.show()


if __name__ == '__main__':
    main()
