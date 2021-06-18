import matplotlib.pyplot as plt
import numpy as np
from joblib import dump
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from src.utils.read_data import read_data
from statistical_feature_extraction import FeatureExtraction

data_files, _ = read_data()
feature_class = FeatureExtraction()
point_wise_features = np.asarray([feature_class.point_wise_extraction(file) for file in data_files])
point_wise_column_number = point_wise_features.shape[1] - 1
point_wise_data_train, point_wise_data_test, point_wise_labels_train, point_wise_labels_test = \
    train_test_split(point_wise_features[:, :point_wise_column_number], point_wise_features[:, point_wise_column_number]
                     , test_size=0.3, random_state=42)
print(point_wise_labels_train, point_wise_labels_test)
classifier_1 = svm.LinearSVC()  # One Vs. Rest
classifier_2 = RandomForestClassifier(random_state=0)

classifier_1.fit(point_wise_data_train, point_wise_labels_train)
classifier_2.fit(point_wise_data_train, point_wise_labels_train)

predicted_labels_1 = classifier_1.predict(point_wise_data_test)
predicted_labels_2 = classifier_2.predict(point_wise_data_test)
confusion_matrix_1 = confusion_matrix(point_wise_labels_test, predicted_labels_1, labels=["1", "2", "3"])
confusion_matrix_2 = confusion_matrix(point_wise_labels_test, predicted_labels_2, labels=["1", "2", "3"])
print("Point Wise Features & SVM 1 Vs. Rest")
print(confusion_matrix_1)
print(f"Accuracy {accuracy_score(point_wise_labels_test, predicted_labels_1)} ")
print("Point Wise Features & RF")
print(confusion_matrix_2)
print(f"Accuracy {accuracy_score(point_wise_labels_test, predicted_labels_2)} ")
display_1 = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_1, display_labels=["gesture1", "gesture2", "gesture3"])
display_2 = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_2, display_labels=classifier_2.classes_)
display_1.plot()
display_2.plot()
plt.show()
dump(classifier_1, 'linearSVMClassifier.joblib')
dump(classifier_2, 'RandomForest.joblib')
