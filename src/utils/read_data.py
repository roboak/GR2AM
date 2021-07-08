import ast
import os
import re
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.utils.dataclass import Data


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
            # Recording the wrist coordinate of the first frame of each sequence.
            if i == 0:
                reference_x = df["X"][0]
                reference_y = df["Y"][0]
                reference_z = df["Z"][0]
            df["X"] = df["X"] - reference_x
            df["X"] = df["X"] - df["X"].mean()
            df["Y"] = df["Y"] - reference_y
            df["Y"] = df["Y"] - df["Y"].mean()
            df["Z"] = df["Z"] - reference_z
            df["Z"] = df["Z"] - df["Z"].mean()

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
