from dataclasses import dataclass
import numpy as np


@dataclass
class Data:
    data: np.array
    label: str
