# TODO: Write the function to read the data from gesture capturing and return it to lstm
import pandas as pd
import numpy as np
import ast
from os.path import dirname, abspath
import os
from pathlib import Path
from dataclass import Data
import re


def read_data():
    # From the current file get the parent directory and create a purepath to the Dataset folder
    parent_directory = Path(dirname(dirname(abspath(__file__))))
    path = parent_directory / "HandDataset"
    # List all file names ending with .txt
    file_names = [file for file in os.listdir(str(path)) if file.endswith(".txt")]
    data_list = []

    for file_name in file_names:
        # open the file
        with open(str(path / file_name), 'r') as file:
            # Get the correct label based on the file name
            label = re.search(r'gesture(\d+)_\d+.txt', file_name).group(1)
            # Read all frames
            dataframes = file.readlines()

        empty_list = []
        # Convert the str represented list to an actual list again
        for frame in dataframes:
            frame = ast.literal_eval(frame)
            df = pd.DataFrame(frame)
            empty_list.append(df)
        # Input into a NP array
        data_array = np.asarray(empty_list)
        # Create a data class combining data and label
        data_1 = Data(data=data_array, label=label)

        # save the list for each capture
        data_list.append(data_1)
    return data_list
