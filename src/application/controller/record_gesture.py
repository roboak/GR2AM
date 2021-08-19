import json
import os.path
import platform
from os.path import abspath, dirname
from pathlib import Path

from flask import Blueprint, Response, flash, session

from application.config import config
from dl.deep_learning_model import DeepLearningClassifier
from gesture_capturing import GestureCapture
from machine_learning_working.machine_learning_model import MachineLearningClassifier
from utils.dataclass import GestureMetaData
from pynput.keyboard import Key, Controller

keyboard = Controller()
bp = Blueprint("record_gesture", __name__)

"""API to display video in the record gesture webpage."""


@bp.route('/video_feed/<gesture_name>')
def video_feed(gesture_name):
    parent_directory = dirname(dirname(dirname(dirname(abspath(__file__)))))
    parent_directory = Path(parent_directory)
    path = parent_directory / config.GESTURE_FOLDER_NAME / session['username']
    if not os.path.isdir(path):
        os.mkdir(path)
    gestureMetaData = GestureMetaData(gesture_name=gesture_name)

    # Source: https://towardsdatascience.com/video-streaming-in-web-browsers-with-opencv-flask-93a38846fe00
    if platform.system() == "Darwin":  # for mac input 1 is the camera
        gesture = GestureCapture(folder_location=str(path), gesture_meta_data=gestureMetaData, camera_input_value=1)
    else:
        gesture = GestureCapture(folder_location=str(path), gesture_meta_data=gestureMetaData, camera_input_value=0)
    return Response(gesture.get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


@bp.route('/remove-gesture/<gesture_id>', methods=['GET'])
def removeGesture(gesture_id):
    parent_directory = dirname(dirname(dirname(dirname(abspath(__file__)))))
    parent_directory = Path(parent_directory)
    path = parent_directory / config.GESTURE_FOLDER_NAME / session['username']
    if not os.path.isdir(path):
        return Response(status=500)

    # look for all files beginning with gesture_id (due to the numbering)
    files = [file for file in os.listdir(path) if str.startswith(file, gesture_id)]

    # delete all those files
    for f in files:
        os.remove(path / f)
        print("Removing", f)

    if os.path.isfile(path / 'MetaData.json'):
        with open(str(path / 'MetaData.json'), 'r+') as metafile:
            gesture_dict = json.load(metafile)
            del gesture_dict[gesture_id]
            json.dump(gesture_dict, metafile)  # FIXME this appends instead of overwrite!

    flash("Removed " + gesture_id + " successfully")

    return Response(status=200)


@bp.route('/nextClick', methods=['GET'])
def nextClick():
    keyboard.press('n')
    keyboard.release('n')

    return Response(status=200)


@bp.route('/recordClick', methods=['GET'])
def recordClick():
    keyboard.press("s")
    keyboard.release("s")

    return Response(status=200)


@bp.route('/redoClick', methods=['GET', 'POST'])
def redoClick():
    keyboard.press('r')
    keyboard.release('r')

    return Response(status=200)


@bp.route('/generate_model')
def generate_model():
    """API to generate model"""
    parent_directory = dirname(dirname(dirname(dirname(abspath(__file__)))))
    parent_directory = Path(parent_directory)
    path = parent_directory / config.GESTURE_FOLDER_NAME

    try:
        ml = MachineLearningClassifier(training_data_path=path, training_data_folder=session['username'], window_size=30)
        ml.save_model()

        dl = DeepLearningClassifier(window_size=30, model=None, output_size=18)
        dl.train_model(model_path=str(path / session['username'])+'/state_dict.pt', path_to_data=path, folder_name=session['username'])
    except:
        flash("Error generating model", 'error')
        return Response(status=500)
    else:
        flash("Model was successfully generated")
        return Response(response='Model successfully generated', status=200)
