import numpy as np
import tsfel
from scipy.stats import iqr, kurtosis, skew, median_abs_deviation
from numpy import mean, std, median, var


class FeatureExtraction:
    def __init__(self):
        # TODO: Remove this stuff if agreed on the new extraction methods
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
        label = int(file.label)
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
        :return: an array of 8 statistical features
                ['Interquartile range', 'Kurtosis',
                'Mean', 'Median', 'Median absolute deviation',
                'Skewness', 'Standard deviation', 'Variance']

        """
        array_iqr = iqr(array)
        array_kurtosis = kurtosis(array)
        array_skew = skew(array)
        array_median_abs_deviation = median_abs_deviation(array)
        array_mean = mean(array)
        array_std = std(array)
        array_median = median(array)
        array_var = var(array)
        result = [array_iqr, array_kurtosis, array_skew, array_median_abs_deviation, array_mean, array_std,
                  array_median, array_var]
        # result = tsfel.time_series_features_extractor(self.feature_config_file, array)
        return result
