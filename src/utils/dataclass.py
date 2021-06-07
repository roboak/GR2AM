from dataclasses import dataclass

import numpy as np


@dataclass
class Data:
    data: np.array
    label: str


class GestureMetaData:
    def __init__(self, gesture_name):
        self.gestureName = gesture_name
        self.trials = 0
        self.files = []
