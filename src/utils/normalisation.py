

from scipy.spatial import distance

class Normalisation:
    base_scale = 0.12
    @staticmethod
    def normalise_scale(hand_data):
        point_5 = (hand_data['X'][5], hand_data['Y'][5], hand_data['Z'][5])  # Index_mcp
        point_17 = (hand_data['X'][17], hand_data['Y'][17], hand_data['Z'][17])  # pinky_mcp
        distance_5_17 = distance.euclidean([point_5[0], point_5[1], point_5[2]], [point_17[0], point_17[1], point_17[2]])
        scale_factor = Normalisation.base_scale / distance_5_17
        for _, row in hand_data.iterrows():
            row['X'] = row['X'] * scale_factor
            row['Y'] = row['Y'] * scale_factor
            row['Z'] = row['Z'] * scale_factor
        return hand_data

    @staticmethod
    def normalise_coordinates(df, reference_coord):
        # Recording the wrist coordinate of the first frame of each sequence.
        df["X"] = df["X"] - reference_coord[0]
        df["X"] = df["X"] - df["X"].mean()
        df["Y"] = df["Y"] - reference_coord[1]
        df["Y"] = df["Y"] - df["Y"].mean()
        df["Z"] = df["Z"] - reference_coord[2]
        df["Z"] = df["Z"] - df["Z"].mean()
        return df


    @staticmethod
    def normalize_data(hand_data, reference_coord):
        return Normalisation.normalise_coordinates(Normalisation.normalise_scale(hand_data), reference_coord)
