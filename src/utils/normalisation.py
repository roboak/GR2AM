from scipy.spatial import distance


class Normalisation:
    base_scale = 0.08

    @staticmethod
    def normalise_scale(hand_data):
        point_5 = (hand_data['X'][5], hand_data['Y'][5])  # Index_mcp
        point_17 = (hand_data['X'][17], hand_data['Y'][17])  # pinky_mcp
        distance_5_17 = distance.euclidean([point_5[0], point_5[1]], [point_17[0], point_17[1]])
        scale_factor = Normalisation.base_scale / distance_5_17
        for _, row in hand_data.iterrows():
            row['X'] = row['X'] * scale_factor
            row['Y'] = row['Y'] * scale_factor
            row['Z'] = row['Z'] * scale_factor
        return hand_data

    @staticmethod
    def normalise_coordinates(hand_data, reference_coord):
        # Recording the wrist coordinate of the first frame of each sequence.
        hand_data["X"] = hand_data["X"] - reference_coord[0]
        hand_data["X"] = hand_data["X"] - hand_data["X"].mean()
        hand_data["Y"] = hand_data["Y"] - reference_coord[1]
        hand_data["Y"] = hand_data["Y"] - hand_data["Y"].mean()
        hand_data["Z"] = hand_data["Z"] - reference_coord[2]
        hand_data["Z"] = hand_data["Z"] - hand_data["Z"].mean()
        return hand_data

    @staticmethod
    def normalize_data(hand_data, reference_coord):
        return Normalisation.normalise_coordinates(Normalisation.normalise_scale(hand_data), reference_coord)
