import platform
from os.path import abspath, dirname
from pathlib import Path

from flask import Blueprint, Response

from application.config import config
from dl.deep_learning_model import DeepLearningClassifier
from gesture_capturing import GestureCapture
from utils.dataclass import GestureMetaData
from pynput.keyboard import Key, Controller

keyboard = Controller()
bp = Blueprint("record_gesture", __name__)

"""API to display video in the record gesture webpage."""


@bp.route('/video_feed/<gesture_name>')
def video_feed(gesture_name):
    parent_directory = dirname(dirname(dirname(dirname(abspath(__file__)))))
    parent_directory = Path(parent_directory)
    path = parent_directory / config.GESTURE_FOLDER_NAME
    gestureMetaData = GestureMetaData(gesture_name=gesture_name)

    # Source: https://towardsdatascience.com/video-streaming-in-web-browsers-with-opencv-flask-93a38846fe00
    if platform.system() == "Darwin":  # for mac input 1 is the camera
        gesture = GestureCapture(folder_location=str(path), gesture_meta_data=gestureMetaData, camera_input_value=1)
    else:
        gesture = GestureCapture(folder_location=str(path), gesture_meta_data=gestureMetaData, camera_input_value=0)
    return Response(gesture.get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')






@bp.route('/nextClick', methods=['GET', 'POST'])
def nextClick():
    keyboard.press('n')
    keyboard.release('n')

    return Response(status=200)


@bp.route('/recordClick', methods=['GET', 'POST'])
def recordClick():
    keyboard.press("s")
    keyboard.release("s")

    return Response(status=200)


@bp.route('/redoClick', methods=['GET', 'POST'])
def redoClick():
    keyboard.press('r')
    keyboard.release('r')

    return Response(status=200)

"""API to generate model"""
@bp.route('/generate_model')
def generate_model():
    dl = DeepLearningClassifier()
    dl.train_model()
