import json
import multiprocessing
import threading
from multiprocessing import Queue
from os.path import abspath, dirname
from pathlib import Path
from sys import platform

from flask import Blueprint, Response, flash, redirect, render_template, request, session, url_for

import main
from frontend.config import config
from learning_models.classifying import Classify
from use_case.gesture_capturing import GestureCapture
from utils.gesture_preprocessing.windowing import Windowing

bp = Blueprint("home_page", __name__)

"""This function returns the gestures from "all_gestures.json" file which 
are not present  in the captured_gestures.json file"""


def get_unrecorded_gestures():
    with open("static/js/all_gestures.json") as jsonFile:
        all_gestures = json.load(jsonFile)
        jsonFile.close()

    captured_gestures = {}
    if 'username' in session:
        with open("static/js/" + session["username"] + "/captured_gestures.json") as jsonFile:
            captured_gestures = json.load(jsonFile)
            jsonFile.close()
    unrecorded_gestures = {key: all_gestures[key] for key in all_gestures if key not in captured_gestures}

    return unrecorded_gestures


"""Returns the page that records gestures from the user."""


@bp.route("/add_gesture")
def add_gesture():
    if 'username' not in session:
        return redirect(url_for('login'))

    with open("static/js/" + session["username"] + "/captured_gestures.json") as jsonFile:
        captures = json.load(jsonFile)
        jsonFile.close()

    unrecorded_gestures = get_unrecorded_gestures()
    return render_template("generating_model_capturing_data.html", captures=captures, gestures=unrecorded_gestures)


@bp.route('/gesture_application_mapping', methods=['POST'])
def add_gesture_application_mapping():
    """
    Take an object containing a map of gesture and frontend
    Update the gesture_application_mapping.json file - Add an entry for the input gesture and frontend. If the entry already exists
    :return:
    Return home page with updated gesture_application_mapping.json file.
    """

    gesture_id = request.json.get('gesture_id')
    gesture_name = request.json.get('gesture_name')
    app_id = request.json.get('app_id')
    app_name = request.json.get('app_name')

    print(gesture_id, gesture_name, app_id, app_name)

    if 'username' in session:
        with open("static/js/" + session["username"] + "/gesture_application_mapping.json", "r") as jsonFile:
            mappings = json.load(jsonFile)
            jsonFile.close()
        mappings[gesture_id] = [gesture_name, {app_id: app_name}]

        with open("static/js/" + session["username"] + "/gesture_application_mapping.json", "w") as jsonFile:
            json.dump(mappings, jsonFile)
            jsonFile.close()

        flash("Mapping updated")
        return Response(status=200)
    else:
        flash("An error occurred", 'error')
        return Response(status=401)


def concFunc(sess):
    parent_directory = Path(dirname(dirname(dirname(dirname(abspath(__file__))))))
    path = parent_directory / config.GESTURE_FOLDER_NAME / sess

    main.main(["-l", "-m", str(path)])


@bp.route('/start')
def start_app():

    # th = threading.Thread(target=concFunc, args=(session['username'],))
    th = multiprocessing.Process(target=concFunc, args=(session['username'],))
    th.start()

    return Response(status=200)
