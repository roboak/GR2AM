import json
import os
import shutil
from os.path import abspath, dirname
from pathlib import Path

from flask import Flask, Response, flash, redirect, render_template, request, session, url_for

from frontend.config import config
from frontend.controller import home_page, record_gesture

app = Flask(__name__)
app.register_blueprint(record_gesture.bp)
app.register_blueprint(home_page.bp)

app.secret_key = b'DUMMYKEY'

"""API that returns the index page with captured getures,
 available applications and application_gesture mapping"""


@app.route("/", methods=['GET', 'POST'])
def index():
    if 'username' in session:
        init()
    else:
        return redirect(url_for('login'))

    if request.method == 'POST':
        pass
    elif request.method == 'GET':
        pass

    captured_gestures, mappings = {}, {}
    if 'username' in session:
        # return f'Logged in as {session["username"]}'
        with open("static/js/" + session["username"] + "/captured_gestures.json") as jsonFile:
            captured_gestures = json.load(jsonFile)
            jsonFile.close()

        with open("static/js/" + session["username"] + "/gesture_application_mapping.json") as jsonFile:
            mappings = json.load(jsonFile)
            jsonFile.close()

    with open("static/js/available_applications.json") as jsonFile:
        apps = json.load(jsonFile)
        jsonFile.close()

    return render_template("home_page.html", gestures=captured_gestures, mappings=mappings, apps=apps)


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")
    if request.method == 'POST':
        session['username'] = request.form['username']

        # Check if user was already created once by checking for folder, else create new folder from existing files
        if os.path.isdir('static/js/' + session['username']):  # user already exists
            flash("Login successful, user loaded")
            pass
        else:  # new user -> create user folder + copy all files over
            os.mkdir('static/js/' + session['username'])
            files = [file for file in os.listdir("static/js") if str.endswith(file, ".json")]

            for file_name in files:
                full_file_name = os.path.join("static/js", file_name)
                if os.path.isfile(full_file_name):
                    shutil.copy(full_file_name, "static/js/" + session['username'])

            flash("Login successful, new user created")

        return redirect(url_for('index'))


@app.route('/logout')
def logout():
    session.pop('username', None)
    flash("Logout successful")
    return redirect(url_for('index'))


def init():
    """Read the folder where the gestures are recorded and accordingly update the
    captured_gestures.json file."""

    parent_directory = dirname(dirname(dirname(abspath(__file__))))
    parent_directory = Path(parent_directory)
    path = parent_directory / config.GESTURE_FOLDER_NAME / session['username'] / "MetaData.json"
    if not os.path.isfile(path):
        print("Nothing there yet")
        return

    with open(path, "r") as jsonFile:
        recorded_gestures_dict = json.load(jsonFile)
        jsonFile.close()

    recorded_gesture_names = list(recorded_gestures_dict.keys())

    if 'username' in session:
        with open("static/js/" + session["username"] + "/all_gestures.json", "r") as jsonFile:
            all_gestures = json.load(jsonFile)
            jsonFile.close()

        gestures_gifs = dict()
        for ele in recorded_gesture_names:
            if recorded_gestures_dict[ele]["trials"] > config.THRESHOLD_TRIALS:
                if len(all_gestures[ele]) == 3:
                    gestures_gifs[ele] = [all_gestures[ele][0], all_gestures[ele][1], all_gestures[ele][2]]
                else:
                    gestures_gifs[ele] = [all_gestures[ele][0], all_gestures[ele][1], False]

        with open("static/js/" + session["username"] + "/captured_gestures.json", "w") as jsonFile:
            json.dump(gestures_gifs, jsonFile)
            jsonFile.close()


if __name__ == "__main__":
    app.run(debug=True)
