import ast
import os
import re
from os.path import abspath, dirname
from pathlib import Path

import numpy as np
import pandas as pd

from dataclass import Data
import cv2


def normalize_data(column):
    # Different Normalization Methods https://www.originlab.com/doc/X-Function/ref/rnormalize#Algorithm
    minimum = column.min()
    maximum = column.max()
    column = (column - minimum) / (maximum - minimum)
    return column


def read_data() -> list:
    # From the current file get the parent directory and create a purepath to the Dataset folder
    height, width, channels = cv2.imread("sample.jpg").shape
    parent_directory = Path(dirname(dirname(abspath(__file__))))
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
            label = re.search(r'gesture(\d+)_\d+.txt', file_name).group(1)
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
            df["X"] = normalize_data(df["X"]*width)
            df["Y"] = normalize_data(df["Y"]*height)
            empty_list.append(df)

        # pad all with zeros to the largest size
        while len(empty_list) < largest_frame_count:
            empty_list.append(pd.DataFrame(np.zeros((21, 3))))

        # Input into a NP array
        data_array = np.asarray(empty_list)
        # Create a data class combining data and label
        data_1 = Data(data=data_array, label=label)

        # save the list for each capture
        data_list.append(data_1)

    return data_list
