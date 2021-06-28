from src.machine_learning_working.get_classifier import Classifier
from src.machine_learning_working.predict import Predict

classifier_class = Classifier(training_data_path="", training_data_folder="")
classifier = classifier_class.get_classifier_loaded("training_features_josh.joblib")
predict_class = Predict(classifier=classifier)
# predict_class.predict_data(file_data)
