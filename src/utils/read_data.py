import ast
import os
import re
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.spatial import distance

from src.utils.dataclass import Data


def return_scaled_hand_cordinates(x, y):
    return int(x*1280), int(y*720)


base_scale = 65
def normalize_scale(hand_data):
    # calculate distance between outer most metacarpals and scale with factor
    point_5 = return_scaled_hand_cordinates(hand_data['X'][5], hand_data['Y'][5])  # Index_mcp
    point_17 = return_scaled_hand_cordinates(hand_data['X'][17], hand_data['Y'][17])  # pinky_mcp
    distance_5_17 = distance.euclidean([point_5[0], point_5[1]], [point_17[0], point_17[1]])
    scale_factor = base_scale / distance_5_17

    reference_x = hand_data['X'][0]   # - 0.5  # (image.shape[1] / 2)
    reference_y = hand_data['Y'][0]   # - 0.5  # (image.shape[0] / 2)
    for _, row in hand_data.iterrows():
        row['X'] = row['X'] * scale_factor
        row['Y'] = row['Y'] * scale_factor
        row['Z'] = row['Z'] * scale_factor
    # TODO: Does it make sense here? I thought to normalize wrt the wrist then translate the whole hand to the middle
        row['X'] = row['X'] - reference_x
        row['Y'] = row['Y'] - reference_y

    reference_x = hand_data['X'][0] - 0.5
    reference_y = hand_data['Y'][0] - 0.5
    reference_z = hand_data['Z'][0] - 0.5
    for _, row in hand_data.iterrows():
        row['X'] = row['X'] - reference_x
        row['Y'] = row['Y'] - reference_y
        row['Z'] = row['Z'] - reference_z

    return hand_data


def read_data(path: str, sub_path="", predef_size=0) -> Tuple[list, int]:
    # From the current file get the parent directory and create a pure path to the Dataset folder
    parent_directory = Path(path)
    path = parent_directory / sub_path
    # List all file names ending with .txt sorted by size
    file_names = [(file, os.path.getsize(path / file)) for file in os.listdir(str(path)) if file.endswith(".txt")]
    file_names.sort(key=lambda file: file[1], reverse=True)
    file_names = [file_name[0] for file_name in file_names]
    data_list = []
    largest_frame_count = None

    for file_name in file_names:
        # open the file
        with open(str(path / file_name), 'r') as file:
            # Get the correct label based on the file name
            label = re.search(r'gesture_._\w+_(\d+)_\d+\.txt', file_name).group(1)
            # Read all frames
            dataframes = file.readlines()

        # storing the largest frame size
        if not (largest_frame_count or predef_size):
            largest_frame_count = len(dataframes)
        elif predef_size:
            largest_frame_count = predef_size

        empty_list = []
        # Convert the str represented list to an actual list again
        for i, frame in enumerate(dataframes):
            frame = ast.literal_eval(frame)
            df = pd.DataFrame(frame)

            df = normalize_scale(df)

            # Recording the wrist coordinate of the first frame of each sequence.
            # if i == 0:
            #     reference_x = df["X"][0]
            #     reference_y = df["Y"][0]
            #     reference_z = df["Z"][0]
            # df["X"] = df["X"] - reference_x
            # df["X"] = df["X"] - df["X"].mean()
            # df["Y"] = df["Y"] - reference_y
            # df["Y"] = df["Y"] - df["Y"].mean()
            # df["Z"] = df["Z"] - reference_z
            # df["Z"] = df["Z"] - df["Z"].mean()

            empty_list.append(df)

        # pad all with zeros to the largest size
        while len(empty_list) < largest_frame_count:
            empty_list.append(pd.DataFrame(np.zeros((21, 3))))

        # This also makes sure longer files are cut down
        empty_list = empty_list[0: largest_frame_count]

        # Input into a NP array
        data_array = np.asarray(empty_list)
        # Create a data class combining data and label
        data_1 = Data(data=data_array, label=label)

        # save the list for each capture
        data_list.append(data_1)

    return data_list, largest_frame_count
