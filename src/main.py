import logging
import os
import platform
import sys
from os.path import abspath, dirname
from pathlib import Path
from multiprocessing import Queue

from classifing import Classify
from dl.deep_learning_model import DeepLearningClassifier
from gesture_capturing import GestureCapture
from machine_learning_working.machine_learning_model import MachineLearningClassifier
from src.utils.dataclass import GestureMetaData

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    os.environ['GLOG_minloglevel'] = '2'

    logging.basicConfig(filename='log.log', level=logging.DEBUG)

    print(sys.argv)

    if sys.argv[1] == "--record" or sys.argv[1] == "-r":

        parent_directory = dirname(dirname(abspath(__file__)))
        parent_directory = Path(parent_directory)
        path = parent_directory / "HandDataset"
        gestureMetaData = GestureMetaData(gesture_name="gesture_n_neg_16")

        if platform.system() == "Darwin":  # for me on mac input 1 is the camera
            gesture = GestureCapture(folder_location=str(path), gesture_meta_data=gestureMetaData, camera_input_value=1)
        else:
            gesture = GestureCapture(folder_location=str(path), gesture_meta_data=gestureMetaData, camera_input_value=0)

        gesture.get_frame()

    elif sys.argv[1] == "--live" or sys.argv[1] == "-l":
        aQueue = Queue()
        bQueue = Queue()


        def startCapture():
            if platform.system() == "Darwin":  # for me on mac input 1 is the camera
                gesture = GestureCapture(camera_input_value=1, aQueue=aQueue, bQueue=bQueue)
            else:
                gesture = GestureCapture(camera_input_value=0)

            gesture.get_frame()


        # t1 = threading.Thread(target=startCapture)
        # t1.start()

        t2 = Classify(aQueue, bQueue)
        t2.start()

        startCapture()

        # t1.join()


    elif sys.argv[1] == "--train" or sys.argv[1] == '-t':  # Train
        #ml = MachineLearningClassifier(training_data_path='../HandDataset', training_data_folder='Josh')
        #ml.save_features()  # FIXME change to save_model and change inside hybrid model class!!

        dl = DeepLearningClassifier()
        dl.train_model()

    else:
        print("Please use -r to record or -l for live classification")
