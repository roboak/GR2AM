import json
import os.path
import platform
from os.path import abspath, dirname
from pathlib import Path

from flask import Blueprint, Response, flash, send_file, session
from pynput.keyboard import Controller

from frontend.config import config
from learning_models.machine_learning_model.machine_learning_model import MachineLearningClassifier
from learning_models.neural_network_model.deep_learning_model import DeepLearningClassifier
from use_case.gesture_capturing import GestureCapture
from utils.gesture_data_related.dataclass import GestureMetaData

keyboard = Controller()
bp = Blueprint("record_gesture", __name__)

"""API to display video in the record gesture webpage."""


def get_session_path():
    parent_directory = dirname(dirname(dirname(dirname(abspath(__file__)))))
    parent_directory = Path(parent_directory)
    path = parent_directory / config.GESTURE_FOLDER_NAME / session['username']
    return path


def get_no_custom_gestures():
    """
    Get the current number of custom gestures in use.
    :return: Number of custom gestures as int
    """
    path = get_session_path()

    if os.path.isdir(path) and os.path.isfile(path / 'MetaData.json'):
        with open(str(path / 'MetaData.json'), 'r') as metafile:
            content = ''.join(metafile.readlines())
            return content.count('gesture_c_cust_') // 2

    return 0


@bp.route('/video_feed/<gesture_name>')
def video_feed(gesture_name: str):

    # Handle custom gesture naming
    if gesture_name.startswith("custom_"):
        next_id = get_no_custom_gestures()
        if next_id <= 5:
            ext_name = gesture_name[7:]
            gesture_name = 'gesture_c_cust_1' + str(next_id)

            all_gestures = None
            with open("static/js/" + session["username"] + "/all_gestures.json", "r") as jsonFile:
                all_gestures = json.load(jsonFile)

                all_gestures[gesture_name] = ['', ext_name]

                jsonFile.close()

            with open("static/js/" + session["username"] + "/all_gestures.json", "w") as jsonFile:
                json.dump(all_gestures, jsonFile)
                jsonFile.close()
        else:
            flash("Max number of custom gestures reached")
            return Response(status=405)

    # Handle new folder -> copy negative classes
    path = get_session_path()
    if not os.path.isdir(path):
        os.mkdir(path)

        path = path.parent

        # Copy over neg class files
        files = [file for file in os.listdir(path / "NegativeClasses") if str.endswith(file, ".txt")]

        for file_name in files:
            full_file_name = os.path.join(path / "NegativeClasses", file_name)
            if os.path.isfile(full_file_name):
                print(full_file_name, file_name)
                os.symlink(full_file_name, path / session['username'] / file_name)

        path = path / session['username']

    # Handle gesture recording
    gestureMetaData = GestureMetaData(gesture_name=gesture_name)

    # Source: https://towardsdatascience.com/video-streaming-in-web-browsers-with-opencv-flask-93a38846fe00
    if platform.system() == "Darwin":  # for mac input 1 is the camera
        gesture = GestureCapture(folder_location=str(path), gesture_meta_data=gestureMetaData, camera_input_value=1,
                                 frontend=True)
    else:
        gesture = GestureCapture(folder_location=str(path), gesture_meta_data=gestureMetaData, camera_input_value=0,
                                 frontend=True)
    return Response(gesture.get_frame_yield(), mimetype='multipart/x-mixed-replace; boundary=frame')


@bp.route('/progress/<gesture_name>')
def gesture_progress(gesture_name):
    # FIXME does not currently work with custom gestures

    path = get_session_path()

    print(get_no_custom_gestures())

    if not os.path.isdir(path):
        return Response(status=404)

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
    # FIXME does not currently work with custom gestures

    path = get_session_path()
    if not os.path.isdir(path):
        return Response(status=404)

    # look for all files beginning with gesture_id (due to the numbering)
    files = [file for file in os.listdir(path) if str.startswith(file, gesture_id)]

    # delete all those files
    for f in files:
        os.remove(path / f)
        print("Removing", f)

    if os.path.isfile(path / 'MetaData.json'):
        gesture_dict = None
        # Following operation fixes the overwrite/append issue, as to writing less data
        with open(str(path / 'MetaData.json'), 'r') as metafile:
            gesture_dict = json.load(metafile)
            del gesture_dict[gesture_id]
        with open(str(path / 'MetaData.json'), 'w') as metafile:
            json.dump(gesture_dict, metafile)

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
    path = get_session_path()

    try:
        ml = MachineLearningClassifier(training_data_path=path.parent, training_data_folder=session['username'],
                                       window_size=30)
        ml.save_model(save_path=str(path) + '/trained_model.joblib')

        dl = DeepLearningClassifier(window_size=30, model=None, output_size=18)
        dl.train_model(model_path=str(path), path_to_data=path.parent, folder_name=session['username'],
                       img_path=str(path) + "/")
    except Exception as e:
        print(e)
        flash("Error generating model", 'error')
        return Response(status=500)
    else:
        flash("Model was successfully generated")
        return Response(response='Model successfully generated', status=200)


@bp.route('/matrix')
def get_image():
    path = get_session_path()
    matrix = path / 'saved_figure.png'
    if os.path.isfile(matrix):
        return send_file(matrix, mimetype='image/png')

    return Response(status=404)
