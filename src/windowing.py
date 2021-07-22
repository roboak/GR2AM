import copy
import multiprocessing
import time


class Windowing(multiprocessing.Process):
    def __init__(self, aQueue, bQueue, window_size):
        super().__init__()

        self.aQueue = aQueue
        self.bQueue = bQueue
        self.window_size = window_size

        self.all_keypoints = list()
        self.last_append = 0

    def run(self):
        while True:
            if not self.aQueue.empty():  # if there's input data
                self.window_frame(self.aQueue.get())  # call func

            # time condition

    def window_frame(self, frame):
        self.all_keypoints.append(frame)
        self.last_append = time.time()

        # When 60 frames are captured create job to classify
        if len(self.all_keypoints) == self.window_size:
            self.bQueue.put(copy.copy(self.all_keypoints))

            # Record overlapping window
            self.all_keypoints = self.all_keypoints[-int(self.window_size * 0.6):]  # save last x entries for next window


# When 10s from the last frame have passed create job (cond. have at least 21 frames due to overlap)
# if len(self.all_keypoints) > 20 and time.time() >= self.last_append + 10:
#     self.bQueue.put(copy.copy(self.all_keypoints))
#
#     # empty out completely, no related movements
#     self.all_keypoints = []
