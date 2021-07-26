from flask import render_template, Blueprint, request
import json
bp = Blueprint("home_page", __name__)

"""This function returns the gestures from "all_gestures.json" file which 
are not present  in the captured_gestures.json file"""
#FIXME: How to ensure that the available_gesture column is refreshed regularly
def get_unrecorded_gestures():
    with open("static/js/all_gestures.json") as jsonFile:
        all_gestures = json.load(jsonFile)
        jsonFile.close()
    with open("static/js/captured_gestures.json") as jsonFile:
        captured_gestures = json.load(jsonFile)
        jsonFile.close()
    unrecorded_gestures = {key: all_gestures[key] for key in all_gestures if key not in captured_gestures}
    return unrecorded_gestures

"""Returns the page that records gestures from the user."""
@bp.route("/add_gesture")
def add_gesture():

    with open("static/js/captured_gestures.json") as jsonFile:
        captures = json.load(jsonFile)
        jsonFile.close()

    unrecorded_gestures = get_unrecorded_gestures()
    return render_template("generating_model_capturing_data.html", captures=captures, gestures=unrecorded_gestures)

@bp.route('/gesture_application_mapping', methods=['POST'])
def add_gesture_application_mapping():
    """
    Take an object containing a map of gesture and application
    Update the gesture_application_mapping.json file - Add an entry for the input gesture and application. If the entry already exists
    :return:
    Return home page with updated gesture_application_mapping.json file.
    """

    gesture_name = request.json.get('gesture_name')
    application = request.json.get('application')

    with open("static/js/captured_gestures.json") as jsonFile:
        captured_gestures = json.load(jsonFile)
        jsonFile.close()

    with open("static/js/gesture_application_mapping.json", "r") as jsonFile:
        mappings = json.load(jsonFile)
        jsonFile.close()
    mappings[gesture_name] = application
    with open("static/js/gesture_application_mapping.json", "w") as jsonFile:
        json.dump(mappings, jsonFile)
        jsonFile.close()

    with open("static/js/available_applications.json") as jsonFile:
        apps = json.load(jsonFile)
        jsonFile.close()

    # return Response("Successful")
    return render_template("home_page.html", gestures=captured_gestures, mappings=mappings, apps=apps)


