

from scipy.spatial import distance

class Normalisaion:
    base_scale = 0.12
    @staticmethod
    def normalise_scale(hand_data):
        point_5 = (hand_data['X'][5], hand_data['Y'][5])  # Index_mcp
        point_17 = (hand_data['X'][17], hand_data['Y'][17])  # pinky_mcp
        distance_5_17 = distance.euclidean([point_5[0], point_5[1]], [point_17[0], point_17[1]])
        scale_factor = Normalisaion.base_scale / distance_5_17
        for _, row in hand_data.iterrows():
            row['X'] = row['X'] * scale_factor
            row['Y'] = row['Y'] * scale_factor
            row['Z'] = row['Z'] * scale_factor
        return hand_data

    @staticmethod
    def normalise_coordinates(hand_data):
        reference_x = hand_data['X'][0]   # - 0.5  # (image.shape[1] / 2)
        reference_y = hand_data['Y'][0]   # - 0.5  # (image.shape[0] / 2)
        for _, row in hand_data.iterrows():
            row['X'] = row['X'] - reference_x
            row['Y'] = row['Y'] - reference_y
        reference_x = hand_data['X'][0] - 0.5
        reference_y = hand_data['Y'][0] - 0.5
        reference_z = hand_data['Z'][0] - 0.5
        for _, row in hand_data.iterrows():
            row['X'] = row['X'] - reference_x
            row['Y'] = row['Y'] - reference_y
            row['Z'] = row['Z'] - reference_z
            row['X'] -= row['X'].mean()
            row['Y'] -= row['Y'].mean()
            row['Z'] -= row['Z'].mean()
        return hand_data

    @staticmethod
    def normalise_coordinates_1(df, reference_coord):
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
        return Normalisaion.normalise_coordinates_1(Normalisaion.normalise_scale(hand_data), reference_coord)
