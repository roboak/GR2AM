import logging
import os
import platform
import sys
from multiprocessing import Queue
from os.path import abspath, dirname
from pathlib import Path

from src.learning_models.classifying import Classify
from src.learning_models.machine_learning_model.machine_learning_model import MachineLearningClassifier
from src.learning_models.neural_network_model.deep_learning_model import DeepLearningClassifier
from src.use_case.gesture_capturing import GestureCapture
from src.use_case.gesture_application_mapping import ApplicationTriggeringService
from src.utils.gesture_data_related.dataclass import GestureMetaData
# Press the green button in the gutter to run the script.
from src.utils.gesture_preprocessing.windowing import Windowing


def main(args):
    WINDOW_SIZE = 30
    os.environ['GLOG_minloglevel'] = '2'

    logging.basicConfig(filename='log.log', level=logging.DEBUG)

    if args[0] == "--record" or args[0] == "-r":
        userId = ''
        if args[1] != '--user':
            raise Exception("User flag not specified. Add --user <userID>")
        folder = ""
        userId = args[2]
        parent_directory = dirname(dirname(abspath(__file__)))
        parent_directory = Path(parent_directory)


        # fnames_old = ['gesture_t_tap_1', 'gesture_t_up_2', 'gesture_t_down_3', 'gesture_i_tap_4', 'gesture_i_up_5',
        #               'gesture_i_down_6', 'gesture_m_tap_7', 'gesture_m_up_8', 'gesture_m_down_9', 'gesture_r_tap_10',
        #               'gesture_r_up_11', 'gesture_r_down_12', 'gesture_l_tap_13', 'gesture_l_up_14',
        #               'gesture_l_down_15',
        #               'gesture_n_still_16', 'gesture_n_up_17', 'gesture_n_down_18']
        # fnames_new = ['gesture_t_tap_1', 'gesture_t_up_2', 'gesture_t_down_3', 'gesture_i_tap_4', 'gesture_i_up_5',
        #               'gesture_i_down_6', 'gesture_m_tap_7', 'gesture_m_up_8', 'gesture_m_down_9', 'gesture_r_tap_10']
        fnames_still = ['gesture_n_still_16', 'gesture_n_up_17', 'gesture_n_down_18']

        fnames_training = ['gesture_t_tap_1', 'gesture_t_up_2', 'gesture_t_down_3', 'gesture_i_tap_4', 'gesture_i_up_5',
                      'gesture_i_down_6']
        fnames_testing = ['gesture_m_tap_1', 'gesture_m_up_2', 'gesture_m_down_3', 'gesture_r_tap_4', 'gesture_r_up_5',
                      'gesture_r_down_6']
        fnames = []
        if args[3] == '--train_data':
            folder = "train"
            fnames = fnames_training
            print("Recording training data")
        if args[3] == '--eval_data':
            folder = "eval"
            fnames = fnames_testing
            print("Recording testing data")

        # print(fnames)
        path = parent_directory / "HandDataset" / folder  # TODO choose your folder

        for fname in fnames:
            print("Recording {}".format(fname))
            gestureMetaData = GestureMetaData(gesture_name=userId+'_'+fname)

            if platform.system() == "Darwin":  # for me on mac input 1 is the camera
                gesture = GestureCapture(folder_location=str(path), gesture_meta_data=gestureMetaData,
                                         camera_input_value=1)
            else:
                gesture = GestureCapture(folder_location=str(path), gesture_meta_data=gestureMetaData,
                                         camera_input_value=0)

            gesture.get_frame()

    elif args[0] == "--live" or args[0] == "-l":
        print("Starting Live!")

        aQueue = Queue()  # all frames
        bQueue = Queue()  # window
        cQueue = Queue()  # Output of classification
        dQueue = Queue()  # Output of classification will be consumed by a service triggerer

        def startCapture():
            if platform.system() == "Darwin":  # for me on mac input 1 is the camera
                gesture = GestureCapture(camera_input_value=1, aQueue=aQueue, cQueue=cQueue, dQueue=dQueue,
                                         window_size=WINDOW_SIZE)
            else:
                gesture = GestureCapture(camera_input_value=0, aQueue=aQueue, cQueue=cQueue, dQueue=dQueue,
                                         window_size=WINDOW_SIZE)
            gesture.get_frame()

        if len(args) == 4 and (args[1] == '-m' or args[1] == "--model"):
            model_path = args[2] if args[2].endswith("/") else args[2] + "/"

        # Process C
        t = Classify(bQueue=bQueue, cQueue=cQueue, window_size=WINDOW_SIZE,
                     model_path=model_path if 'model_path' in locals() else 'saved_models/')
        t.start()

        # Process B
        t2 = Windowing(aQueue=aQueue, bQueue=bQueue, window_size=WINDOW_SIZE)
        t2.start()

        # Process D- reading classification and triggering applications
        username = args[3]#'test_user'
        t3 = ApplicationTriggeringService(dQueue, username)
        t3.start()
        # Process A
        startCapture()

    elif args[0] == "--train" or args[0] == '-t':  # Train
        # ml = MachineLearningClassifier(training_data_path='../HandDataset', training_data_folder='Abdul_Josh',
        #                                window_size=WINDOW_SIZE)
        # ml.save_model()

        dl = DeepLearningClassifier(window_size=WINDOW_SIZE, model=None, output_size=18)
        dl.train_model(model_path='../HandDataset/Abdul_Josh')

    else:
        print("Please use -r to record or -l for live classification")


if __name__ == '__main__':
    main(sys.argv[1:])
