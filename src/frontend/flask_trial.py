from flask import render_template, Flask, Response
import cv2
import platform
from os.path import abspath, dirname
from pathlib import Path
import json

from src.utils.dataclass import GestureMetaData
from src.gesture_capturing_flask import GestureCapture

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("home_page.html")


@app.route("/add_gesture")
def add_gesture():
    return render_template("generating_model_capturing_data.html")


@app.route('/video_feed')
def video_feed():
    # Source: https://towardsdatascience.com/video-streaming-in-web-browsers-with-opencv-flask-93a38846fe00
    if platform.system() == "Darwin":  # for mac input 1 is the camera
        gesture = GestureCapture(folder_location=str(path), gesture_meta_data=gestureMetaData, camera_input_value=1)
    else:
        gesture = GestureCapture(folder_location=str(path), gesture_meta_data=gestureMetaData, camera_input_value=0)
    return Response(gesture.get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    parent_directory = dirname(dirname(dirname(abspath(__file__))))
    print(parent_directory)
    parent_directory = Path(parent_directory)
    path = parent_directory / "flask_trial_gestures"
    gestureMetaData = GestureMetaData(gesture_name="gesture_1")
    app.run(host="0.0.0.0", port="5000")
