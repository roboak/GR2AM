from flask import render_template, Response, Blueprint, g, redirect, flash, request, url_for
from werkzeug.exceptions import abort
import cv2
import platform
from os.path import abspath, dirname
from pathlib import Path

from utils.dataclass import GestureMetaData
from gesture_capturing import GestureCapture

bp = Blueprint("pre_training", __name__)
parent_directory = dirname(dirname(dirname(abspath(__file__))))
parent_directory = Path(parent_directory)
path = parent_directory / "Akash"

@bp.route("/")
def index():
    return render_template("index1.html", "gesture_1")


"""" Start showing the video"""
@bp.route('/record_gesture/<gesture_name>')
def video_feed(gesture_name):
    """This function should locally store the video file and start streaming the images from backend to GUI"""
    print(gesture_name)
    gestureMetaData = GestureMetaData(gesture_name=gesture_name)
    # https://towardsdatascience.com/video-streaming-in-web-browsers-with-opencv-flask-93a38846fe00
    if platform.system() == "Darwin":  # for mac input 1 is the camera
        gesture = GestureCapture(folder_location=str(path), gesture_meta_data=gestureMetaData, camera_input_value=1)
    else:
        gesture = GestureCapture(folder_location=str(path), gesture_meta_data=gestureMetaData, camera_input_value=0)
    return Response(gesture.get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


"""Start recording the trial for a gesture and save  --> Space Bar"""
# @bp.route('/test')
# def start_recording():
#     """This endpoint should save the gesture recorded in the database"""
#     return Response(gesture.get_frame("record"), mimetype='multipart/x-mixed-replace; boundary=frame')

"""Redo previous trial --> r"""
def redo_trial():
    """This endpoint should be hit to record the next trial for a particular gesture"""
    pass

"""Next trial  --> n"""
def next_trial():
    pass

""" Quit recording trials for that gesture --> q"""
# @bp.route('record-data/<gesture_id>')
@bp.route('/record_data/flag')
def contact():
    if "start_recording" in request.form:
        return("recording started")
    elif "quit_recording" in request.form:
        return("stop recording")




