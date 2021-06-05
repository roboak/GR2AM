import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from statistical_feature_extraction import FeatureExtraction
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, plot_confusion_matrix
from read_data import read_data

data_files = read_data()
feature_class = FeatureExtraction()
point_wise_features = np.asarray([feature_class.point_wise_extraction(file) for file in data_files])
frame_wise_features = np.asarray([feature_class.frame_wise_extraction(file) for file in data_files])
point_wise_column_number = point_wise_features.shape[1] - 1
frame_wise_column_number = frame_wise_features.shape[1] - 1
point_wise_data_train, point_wise_data_test, point_wise_labels_train, point_wise_labels_test = \
    train_test_split(point_wise_features[:, :point_wise_column_number], point_wise_features[:, point_wise_column_number]
                     , test_size=0.3, random_state=42)
frame_wise_data_train, frame_wise_data_test, frame_wise_labels_train, frame_wise_labels_test = \
    train_test_split(frame_wise_features[:, :frame_wise_column_number], frame_wise_features[:, frame_wise_column_number]
                     , test_size=0.3, random_state=42)
print(point_wise_labels_train, point_wise_labels_test)
print(frame_wise_labels_train, frame_wise_labels_test)
clf_1 = svm.SVC(decision_function_shape='ovo')  # One Vs. One
clf_2 = svm.LinearSVC()  # One Vs. Rest
clf_3 = RandomForestClassifier(random_state=0)
clf_4 = svm.SVC(decision_function_shape='ovo')  # One Vs. One
clf_5 = svm.LinearSVC()  # One Vs. Rest
clf_6 = RandomForestClassifier(random_state=0)
clf_1.fit(point_wise_data_train, point_wise_labels_train)
clf_2.fit(point_wise_data_train, point_wise_labels_train)
clf_3.fit(point_wise_data_train, point_wise_labels_train)
clf_4.fit(frame_wise_data_train, frame_wise_labels_train)
clf_5.fit(frame_wise_data_train, frame_wise_labels_train)
clf_6.fit(frame_wise_data_train, frame_wise_labels_train)
predicted_labels_1 = clf_1.predict(point_wise_data_test)
predicted_labels_2 = clf_2.predict(point_wise_data_test)
predicted_labels_3 = clf_3.predict(point_wise_data_test)
predicted_labels_4 = clf_4.predict(frame_wise_data_test)
predicted_labels_5 = clf_5.predict(frame_wise_data_test)
predicted_labels_6 = clf_6.predict(frame_wise_data_test)
print("Point Wise Features & SVM 1 Vs. 1")
print(confusion_matrix(point_wise_labels_test, predicted_labels_1, labels=["1", "2", "3"]))
print(f"Accuracy {accuracy_score(point_wise_labels_test, predicted_labels_1)} ")
print("Point Wise Features & SVM 1 Vs. Rest")
print(confusion_matrix(point_wise_labels_test, predicted_labels_2, labels=["1", "2", "3"]))
print(f"Accuracy {accuracy_score(point_wise_labels_test, predicted_labels_2)} ")
print("Point Wise Features & RF")
print(confusion_matrix(point_wise_labels_test, predicted_labels_3, labels=["1", "2", "3"]))
print(f"Accuracy {accuracy_score(point_wise_labels_test, predicted_labels_3)} ")

print("Frame Wise Features & SVM 1 Vs. 1")
print(confusion_matrix(frame_wise_labels_test, predicted_labels_4, labels=["1", "2", "3"]))
print(f"Accuracy {accuracy_score(frame_wise_labels_test, predicted_labels_4)} ")
print("Frame Wise Features & SVM 1 Vs. Rest")
print(confusion_matrix(frame_wise_labels_test, predicted_labels_5, labels=["1", "2", "3"]))
print(f"Accuracy {accuracy_score(frame_wise_labels_test, predicted_labels_5)} ")
print("Frame Wise Features & RF")
print(confusion_matrix(frame_wise_labels_test, predicted_labels_6, labels=["1", "2", "3"]))
print(f"Accuracy {accuracy_score(frame_wise_labels_test, predicted_labels_6)} ")
