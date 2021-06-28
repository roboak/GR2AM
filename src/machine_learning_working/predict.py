from src.machine_learning_working.statistical_feature_extraction import FeatureExtraction


class Predict:
    def __init__(self, classifier):
        self.classifier = classifier
        self.feature_extraction = FeatureExtraction()

    def predict_data(self, file):
        features = self.feature_extraction.get_features(file)
        column_number = features.shape[1] - 1
        data = features[:, :column_number]
        actual_label = features[:, column_number]
        result = self.classifier.predict(data)
        return result, actual_label
