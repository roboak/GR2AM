import json
import os.path
import platform
from os.path import abspath, dirname
from pathlib import Path

from flask import Blueprint, Response, flash, request, send_file, session

from frontend.config import config
from learning_models.neural_network_model.deep_learning_model import DeepLearningClassifier
from use_case.gesture_capturing import GestureCapture
from learning_models.machine_learning_model.machine_learning_model import MachineLearningClassifier
from utils.gesture_data_related.dataclass import GestureMetaData
from pynput.keyboard import Key, Controller

keyboard = Controller()
bp = Blueprint("record_gesture", __name__)

"""API to display video in the record gesture webpage."""


@bp.route('/video_feed/<gesture_name>')
def video_feed(gesture_name: str):
    if gesture_name.startswith("custom_"):
        gesture_name = 'gesture_c_' + gesture_name[8:] + '_X'

    parent_directory = dirname(dirname(dirname(dirname(abspath(__file__)))))
    parent_directory = Path(parent_directory)
    path = parent_directory / config.GESTURE_FOLDER_NAME / session['username']
    if not os.path.isdir(path):
        os.mkdir(path)

        path = path / ".."

        # Copy over neg class files
        files = [file for file in os.listdir(path / "NegativeClasses") if str.endswith(file, ".txt")]

        for file_name in files:
            full_file_name = os.path.join(path / "NegativeClasses", file_name)
            if os.path.isfile(full_file_name):
                print(full_file_name, file_name)
                os.symlink(full_file_name, path / session['username'] / file_name)

        path = path / session['username']

    gestureMetaData = GestureMetaData(gesture_name=gesture_name)

    # Source: https://towardsdatascience.com/video-streaming-in-web-browsers-with-opencv-flask-93a38846fe00
    if platform.system() == "Darwin":  # for mac input 1 is the camera
        gesture = GestureCapture(folder_location=str(path), gesture_meta_data=gestureMetaData, camera_input_value=1)
    else:
        gesture = GestureCapture(folder_location=str(path), gesture_meta_data=gestureMetaData, camera_input_value=0)
    return Response(gesture.get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


@bp.route('/progress/<gesture_name>')
def gesture_progress(gesture_name):
    # FIXME if gesture_name startwith custom_

    parent_directory = dirname(dirname(dirname(dirname(abspath(__file__)))))
    parent_directory = Path(parent_directory)
    path = parent_directory / config.GESTURE_FOLDER_NAME / session['username']
    if not os.path.isdir(path):
        return Response(status=500)

    if os.path.isfile(path / 'MetaData.json'):
        with open(str(path / 'MetaData.json'), 'r') as metafile:
            gesture_dict = json.load(metafile)
            i = 0
            if gesture_name in gesture_dict:
                i = gesture_dict[gesture_name]['trials'] * 10
            return Response(str(i), mimetype='text/plain', status=200)

    else:  # No file so first ever recording -> start at 0%
        return Response('0', mimetype='text/plain', status=200)


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
        ml.save_model(save_path=str(path / session['username'])+'/trained_model.joblib')

        dl = DeepLearningClassifier(window_size=30, model=None, output_size=18)
        dl.train_model(model_path=str(path / session['username']), path_to_data=path, folder_name=session['username'])
    except Exception:
        flash("Error generating model", 'error')
        return Response(status=500)
    else:
        flash("Model was successfully generated")
        return Response(response='Model successfully generated', status=200)


@bp.route('/matrix')
def get_image():
    parent_directory = dirname(dirname(dirname(dirname(abspath(__file__)))))
    parent_directory = Path(parent_directory)
    path = parent_directory / config.GESTURE_FOLDER_NAME / session['username']
    matrix = path / 'test_plt.png'
    if os.path.isfile(matrix):
        return send_file(matrix, mimetype='image/png')

    return Response(status=404)

