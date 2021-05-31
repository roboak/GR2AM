import pandas as pd
import numpy as np
from dataclasses import dataclass
import ast
from os.path import dirname, abspath
import os
from pathlib import Path


@dataclass
class Data:
    data: np.array
    label: str


parent_directory = dirname(dirname(abspath(__file__)))
print(parent_directory)
parent_directory = Path(parent_directory)
path = parent_directory / "HandDataset"
file_names_dummy = os.listdir(str(path))
file_names = []
for file in file_names_dummy:
    if file.endswith(".txt"):
        file_names.append(file)
print(file_names)
# file = open('../HandDataset/gesture1.txt', 'r')
for file_name in file_names:
    file_path = parent_directory / "HandDataset" / file_name
    file = open(str(file_path), 'r')
    dataframes = file.readlines()
    data_list = []
    for frame in dataframes:
        frame = ast.literal_eval(frame)
        df = pd.DataFrame(frame)
        numpyMatrix = df.to_numpy()
        data_1 = Data(data=numpyMatrix, label=file_name.replace('.txt', ''))  # TODO make label dynamic
        data_list.append(data_1)
    print(data_list)
