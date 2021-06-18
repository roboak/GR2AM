import ast
import os
import re
from os.path import abspath, dirname
from pathlib import Path

import numpy as np
import pandas as pd

from utils.dataclass import Data
import cv2


def read_data() -> tuple[list, int]:
    # From the current file get the parent directory and create a purepath to the Dataset folder
    #height, width, channels = cv2.imread("../sample.jpg").shape
    parent_directory = Path(dirname(dirname(dirname(abspath(__file__)))))
    path = parent_directory / "HandDataset"
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
        if not largest_frame_count:
            largest_frame_count = len(dataframes)

        empty_list = []
        # Convert the str represented list to an actual list again
        for frame in dataframes:
            frame = ast.literal_eval(frame)
            df = pd.DataFrame(frame)

            # FIXME are we normalising correctly here?
            df["X"] = df["X"] - df["X"][0]
            df["X"] = df["X"] - df["X"].mean()
            df["Y"] = df["Y"] - df["Y"][0]
            df["Y"] = df["Y"] - df["Y"].mean()
            # FIXME we we need to normalise the Z coord even?
            df["Z"] = df["Z"] - df["Z"][0]
            df["Z"] = df["Z"] - df["Z"].mean()

            empty_list.append(df)

        # pad all with zeros to the largest size
        while len(empty_list) <= largest_frame_count:
            empty_list.append(pd.DataFrame(np.zeros((21, 3))))
        empty_list = empty_list[0: largest_frame_count]
        # Input into a NP array
        data_array = np.asarray(empty_list)
        # Create a data class combining data and label
        data_1 = Data(data=data_array, label=label)

        # save the list for each capture
        data_list.append(data_1)

    return data_list, largest_frame_count
