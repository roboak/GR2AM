import sys

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix

from dl import CNN1D_Classifier as CNN1D
from machine_learning_working.get_classifier import Classifier
from machine_learning_working.predict import Predict
from utils import format_data_for_nn as ft, read_data as rd


# Load DL model

def dl(dl_model, test_data):
    data = ft.format_individual_data(test_data.data)
    data = torch.from_numpy(data)
    # labels = torch.from_numpy(data_dict["labels"])
    pred = dl_model.forward(data.view(1, 80, 63).float())
    result = (test_data.label, torch.argmax(pred).item() + 1, torch.max(pred).item())
    return result


def ml(classifier, file):
    predict_class = Predict(classifier=classifier)
    result, actual_label = predict_class.predict_data(file)
    return result


if __name__ == '__main__':

    # read data
    test_data, _ = rd.read_data("../HandDataset/TestingData", "Josh", 80)

    # load dl model
    dl_model = CNN1D.CNN1D(80, "cpu", output_size=15)
    dl_model.eval()
    dl_model.load_state_dict(torch.load('model_save/cnn_state_dict_Josh_1.pt'))

    # load features and train the ml model
    classifier_class = Classifier(training_data_path="", training_data_folder="")
    classifier = classifier_class.get_classifier_loaded("./machine_learning_working/training_features_josh.joblib")

    results_dl = []  # actual, predicted, confidence
    results_ml = []
    results_hybrid = []
    labels = []
    for file in test_data:
        result_dl = dl(dl_model, file)
        result_ml = ml(classifier, file)
        results_dl.append(result_dl[1])
        results_ml.append(result_ml[0])
        labels.append(int(result_dl[0]))
        if result_dl[2] < 0.85:
            results_hybrid.append(result_ml[0])
        else:
            results_hybrid.append(int(result_dl[1]))

    confusi_dl = confusion_matrix(labels, results_dl, labels=[x for x in range(1, 16)])
    confusi_ml = confusion_matrix(labels, results_ml, labels=[x for x in range(1, 16)])
    confusi_hybrid = confusion_matrix(labels, results_hybrid, labels=[x for x in range(1, 16)])
    display_dl = ConfusionMatrixDisplay(confusion_matrix=confusi_dl,
                                        display_labels=["g" + str(x) for x in range(1, 16)]).plot()
    display_ml = ConfusionMatrixDisplay(confusion_matrix=confusi_ml,
                                        display_labels=["g" + str(x) for x in range(1, 16)]).plot()
    display_hybrid = ConfusionMatrixDisplay(confusion_matrix=confusi_hybrid,
                                            display_labels=["g" + str(x) for x in range(1, 16)]).plot()

    print(f"Accuracy dl {accuracy_score(labels, results_dl)} ")
    print(f"Accuracy ml {accuracy_score(labels, results_ml)} ")
    print(f"Accuracy hybrid {accuracy_score(labels, results_hybrid)} ")
    plt.show()
