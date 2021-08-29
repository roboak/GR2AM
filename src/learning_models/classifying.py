import multiprocessing
from multiprocessing import Queue
from src.utils.gesture_preprocessing.normalisation import Normalisation as norm
import numpy as np
import pandas as pd
from learning_models.hybrid_learning_model import HybridLearningClassifier


class Classify(multiprocessing.Process):
    def __init__(self, bQueue: Queue, cQueue: Queue, window_size, model_path='saved_models/'):
        super().__init__()
        self.bQueue = bQueue
        self.cQueue = cQueue
        self.window_size = window_size

        # Call learning model to predict class
        self.hl = HybridLearningClassifier(self.window_size, model_path=model_path)

    def run(self):
        while True:
            if not self.bQueue.empty():
                self.classify_capture(self.bQueue.get())

    def classify_capture(self, frames):
        empty_list = []
        reference_x = 0
        reference_y = 0
        reference_z = 0
        # Convert the str represented list to an actual list again
        for i, frame in enumerate(frames):
            df = pd.DataFrame(frame)
            # Recording the wrist coordinate of the first frame of each sequence.
            if i == 0:
                reference_x = df["X"][0]
                reference_y = df["Y"][0]
                reference_z = df["Z"][0]
            df = norm.normalize_data(df, (reference_x, reference_y, reference_z))
            empty_list.append(df)

        # pad all with zeros to the frame size 60
        while len(empty_list) < self.window_size:
            empty_list.append(pd.DataFrame(np.zeros((21, 3))))

        data_array = np.asarray(empty_list)

        data = self.hl.predict_data(data_array)
        self.cQueue.put(data)
        return data
