import numpy as np
import tsfel


class FeatureExtraction:
    def __init__(self):
        self.feature_config_file = tsfel.get_features_by_domain("statistical")
        keys_to_remove = ['ECDF', 'ECDF Percentile', 'ECDF Percentile Count', 'Histogram']
        # These are distributions to describe the data how it's distributed and what it looks like and so on.
        # They aren't used as features.
        for key in keys_to_remove:
            self.feature_config_file["statistical"].pop(key, None)
        return

    def get_features_training(self, file) -> np.array:
        """
        :param file: A file a recording.
        For each file, we have a number of frames.
        For each frame, we have 21 points, each point has 3 coordinates.
        We want to extract the features for each frame for each finger joint by itself.
        :return: A feature array for the file.
        """
        data = file.data
        label = file.label
        feature = np.apply_along_axis(self.__features, 0, data)
        # We have the features calculated for each point
        # The shape = (Number of features, Number of points, Number of coordinates)
        feature = feature.flatten()
        # Will flatten the feature array to be a 1D array
        # The order is feature1 {(x,y,z) for point1, (x,y,z) for point2 .... (x,y,z) for point21}, feature2 and so on
        result = np.append(feature, label)
        return result

    def get_features_prediction(self, data: np.array) -> np.array:
        """
        :param data: A numpy array of just 1 recording
        For each file, we have a number of frames.
        For each frame, we have 21 points, each point has 3 coordinates.
        We want to extract the features for each frame for each finger joint by itself.
        :return: A feature array for the file.
        """
        feature = np.apply_along_axis(self.__features, 0, data)
        # We have the features calculated for each point
        # The shape = (Number of features, Number of points, Number of coordinates)
        features = feature.flatten()
        # Will flatten the feature array to be a 1D array
        # The order is feature1 {(x,y,z) for point1, (x,y,z) for point2 .... (x,y,z) for point21}, feature2 and so on
        # result = np.append(feature)
        return features

    def __features(self, array):
        """
        :param array: 1D numpy array
        :return: an array of 16 statistical features
                ['ECDF', 'ECDF Percentile', 'ECDF Percentile Count', 'Histogram', 'Interquartile range', 'Kurtosis',
                'Max', 'Mean', 'Mean absolute deviation', 'Median', 'Median absolute deviation', 'Min',
                'Root mean square', 'Skewness', 'Standard deviation', 'Variance']
        Each one returns 1 value except 'ECDF' -> 10, 'Histogram' -> 10, ECDF Percentile -> 2,
        'ECDF Percentile Count' -> 2
        """
        return tsfel.time_series_features_extractor(self.feature_config_file, array)
