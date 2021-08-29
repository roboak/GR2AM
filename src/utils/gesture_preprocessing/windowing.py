import copy
import multiprocessing
import time


class Windowing(multiprocessing.Process):
    def __init__(self, aQueue, bQueue, window_size):
        super().__init__()

        self.aQueue = aQueue
        self.bQueue = bQueue
        self.window_size = window_size

        self.all_key_points = list()
        self.last_append = 0

    def run(self):
        while True:
            if not self.aQueue.empty():  # if there's input data
                self.window_frame(self.aQueue.get())  # call func

    def window_frame(self, frame):
        self.all_key_points.append(frame)
        self.last_append = time.time()

        # When WINDOW_SIZE frames are captured create job to classify
        if len(self.all_key_points) == self.window_size:
            self.bQueue.put(copy.copy(self.all_key_points)) # Record overlapping window
            # Save the last x entries for next window
            self.all_key_points = self.all_key_points[-int(self.window_size * 0.6):]
