import logging
import os
import platform
import sys
from multiprocessing import Queue
from os.path import abspath, dirname
from pathlib import Path

from classifing import Classify
from dl.deep_learning_model import DeepLearningClassifier
from gesture_capturing import GestureCapture
from machine_learning_working.machine_learning_model import MachineLearningClassifier
from src.utils.dataclass import GestureMetaData
import service
# Press the green button in the gutter to run the script.
from windowing import Windowing

if __name__ == '__main__':

    WINDOW_SIZE = 30

    os.environ['GLOG_minloglevel'] = '2'

    logging.basicConfig(filename='log.log', level=logging.DEBUG)

    print(sys.argv)

    if sys.argv[1] == "--record" or sys.argv[1] == "-r":

        parent_directory = dirname(dirname(abspath(__file__)))
        parent_directory = Path(parent_directory)
        path = parent_directory / "HandDataset" / "Josh2"  # TODO choose your folder

        fnames_old = ['gesture_t_tap_1', 'gesture_t_up_2', 'gesture_t_down_3', 'gesture_i_tap_4', 'gesture_i_up_5',
                   'gesture_i_down_6', 'gesture_m_tap_7', 'gesture_m_up_8', 'gesture_m_down_9', 'gesture_r_tap_10',
                   'gesture_r_up_11', 'gesture_r_down_12', 'gesture_l_tap_13', 'gesture_l_up_14', 'gesture_l_down_15',
                   'gesture_n_still_16', 'gesture_n_up_17', 'gesture_n_down_18']
        fnames_new = ['gesture_t_tap_1', 'gesture_t_up_2', 'gesture_t_down_3', 'gesture_i_tap_4', 'gesture_i_up_5',
                   'gesture_i_down_6', 'gesture_m_tap_7', 'gesture_m_up_8', 'gesture_m_down_9']
        fnames_still = ['gesture_n_still_16', 'gesture_n_up_17', 'gesture_n_down_18']

        for fname in fnames_new:
            gestureMetaData = GestureMetaData(gesture_name=fname)

            if platform.system() == "Darwin":  # for me on mac input 1 is the camera
                gesture = GestureCapture(folder_location=str(path), gesture_meta_data=gestureMetaData, camera_input_value=1)
            else:
                gesture = GestureCapture(folder_location=str(path), gesture_meta_data=gestureMetaData, camera_input_value=0)

            gesture.get_frame()

    elif sys.argv[1] == "--live" or sys.argv[1] == "-l":
        aQueue = Queue()  # all frames
        bQueue = Queue()  # window
        cQueue = Queue()  # Output of classification
        dQueue = Queue()  # Output of classification will be consumed by a service triggerer


        def startCapture():
            if platform.system() == "Darwin":  # for me on mac input 1 is the camera
                gesture = GestureCapture(camera_input_value=1, aQueue=aQueue, cQueue=cQueue, dQueue= dQueue, window_size=WINDOW_SIZE)
            else:
                gesture = GestureCapture(camera_input_value=0, aQueue=aQueue, cQueue=cQueue, dQueue= dQueue, window_size=WINDOW_SIZE)
            gesture.get_frame()

        # Process C
        t = Classify(bQueue=bQueue, cQueue=cQueue, window_size=WINDOW_SIZE)
        t.start()

        # Process B
        t2 = Windowing(aQueue=aQueue, bQueue=bQueue, window_size=WINDOW_SIZE)
        t2.start()

        # Process D- reading classification and triggering applications
        #t3 = service.Service(dQueue)
        #t3.start()
        # Process A
        startCapture()

    elif sys.argv[1] == "--train" or sys.argv[1] == '-t':  # Train
        ml = MachineLearningClassifier(training_data_path='../HandDataset', training_data_folder='Abdul_Josh', window_size=WINDOW_SIZE)
        ml.save_model()

        dl = DeepLearningClassifier(window_size=WINDOW_SIZE, model=None, output_size=18)
        dl.train_model(model_path='../HandDataset/Abdul_Josh')

    else:
        print("Please use -r to record or -l for live classification")
