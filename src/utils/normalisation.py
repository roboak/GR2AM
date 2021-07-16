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
    def normalise_coordinates(hand_data):
        hand_data['X'] -= hand_data['X'][0]  # - 0.5  # (image.shape[1] / 2)
        hand_data['Y'] -= hand_data['Y'][0]  # - 0.5  # (image.shape[0] / 2)

        hand_data["X"] -= hand_data['X'][0] - 0.5
        hand_data["Y"] -= hand_data['Y'][0] - 0.5
        hand_data["Z"] -= hand_data['Z'][0] - 0.5

        hand_data["X"] -= hand_data["X"].mean()
        hand_data["Y"] -= hand_data["Y"].mean()
        hand_data["Z"] -= hand_data["Z"].mean()
        return hand_data

    reference_x = 0
    reference_y = 0
    reference_z = 0

    @staticmethod
    def normalise_coordinates_1(hand_data):
        # Recording the wrist coordinate of the first frame of each sequence.
        if not (Normalisation.reference_x and Normalisation.reference_y and Normalisation.reference_z):
            Normalisation.reference_x = hand_data["X"][0]
            Normalisation.reference_y = hand_data["Y"][0]
            Normalisation.reference_z = hand_data["Z"][0]

        hand_data["X"] -= Normalisation.reference_x
        hand_data["Y"] -= Normalisation.reference_y
        hand_data["Z"] -= Normalisation.reference_z

        # FIXME is the hand in the right position or do we need to subtract 0.5 again?

        hand_data["X"] -= hand_data["X"].mean()
        hand_data["Y"] -= hand_data["Y"].mean()
        hand_data["Z"] -= hand_data["Z"].mean()

        return hand_data

    @staticmethod
    def normalize_data(hand_data):
        return Normalisation.normalise_coordinates_1(Normalisation.normalise_scale(hand_data))
