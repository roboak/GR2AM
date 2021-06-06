import ast
import os
import re
from os.path import abspath, dirname
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from dataclass import Data
from statistical_feature_extraction import FeatureExtraction


def read_data(folder_name: str, sub_folder_name: str) -> list:
    # From the current file get the parent directory and create a pure path to the Dataset folder
    height, width, channels = cv2.imread("sample.jpg").shape
    parent_directory = Path(dirname(dirname(abspath(__file__))))
    path = parent_directory / folder_name / sub_folder_name
    # List all file names ending with .txt sorted by size
    file_names = [(file, os.path.getsize(path / file)) for file in os.listdir(str(path)) if file.endswith(".txt")]
    file_names.sort(key=lambda file: file[1], reverse=True)
    file_names = [file_name[0] for file_name in file_names]
    data_list = []
    largest_frame_count = None

    for file_name in file_names:
        # open the file
        with open(str(path / file_name), 'r') as file:
            # Get the correct label based on the file name
            label = re.search(r'gesture(\d+)_\d+.txt', file_name).group(1)
            # Read all frames
            dataframes = file.readlines()

        # storing the largest frame size
        if not largest_frame_count:
            largest_frame_count = len(dataframes)

        empty_list = []
        # Convert the str represented list to an actual list again
        for frame in dataframes:
            frame = ast.literal_eval(frame)
            df = pd.DataFrame(frame)
            df["X"] = df["X"] - df["X"][0]
            df["Y"] = df["Y"] - df["Y"][0]
            empty_list.append(df)

        # pad all with zeros to the largest size
        while len(empty_list) < largest_frame_count:
            empty_list.append(pd.DataFrame(np.zeros((21, 3))))

        # Input into a NP array
        data_array = np.asarray(empty_list)
        # Create a data class combining data and label
        data_1 = Data(data=data_array, label=label)

        # save the list for each capture
        data_list.append(data_1)

    return data_list


# classifier_1 = load("linearSVMClassifier.joblib")
# classifier_2 = load("RandomForest.joblib")
classifier_1 = svm.LinearSVC()  # One Vs. Rest
classifier_2 = RandomForestClassifier(random_state=0)
files_training = read_data(folder_name="OneParticipantDataSet", sub_folder_name="training")
files_testing = read_data(folder_name="OneParticipantDataSet", sub_folder_name="testing")
feature_class = FeatureExtraction()
features_training = np.asarray([feature_class.point_wise_extraction(file) for file in files_training])
features_testing = np.asarray([feature_class.point_wise_extraction(file) for file in files_testing])
column_numbers = features_training.shape[1] - 1
# cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=0)
# scores_1 = cross_val_score(classifier_1, features[:, :column_numbers], features[:, column_numbers], cv=5)
# scores_2 = cross_val_score(classifier_2, features[:, :column_numbers], features[:, column_numbers], cv=5)
# print("%0.2f accuracy with a standard deviation of %0.2f" % (scores_1.mean(), scores_1.std()))
# print("%0.2f accuracy with a standard deviation of %0.2f" % (scores_2.mean(), scores_2.std()))
classifier_1.fit(features_training[:, :column_numbers], features_training[:, column_numbers])
classifier_2.fit(features_training[:, :column_numbers], features_training[:, column_numbers])
predicted_labels_1 = classifier_1.predict(features_testing[:, :column_numbers])
predicted_labels_2 = classifier_2.predict(features_testing[:, :column_numbers])
confusion_matrix_1 = confusion_matrix(features_testing[:, column_numbers], predicted_labels_1, labels=["1", "2", "3"])
confusion_matrix_2 = confusion_matrix(features_testing[:, column_numbers], predicted_labels_2, labels=["1", "2", "3"])
print("Point Wise Features & SVM 1 Vs. Rest")
print(confusion_matrix_1)
print(f"Accuracy {accuracy_score(features_testing[:, column_numbers], predicted_labels_1)} ")
print("Point Wise Features & RF")
print(confusion_matrix_2)
print(f"Accuracy {accuracy_score(features_testing[:, column_numbers], predicted_labels_2)} ")
