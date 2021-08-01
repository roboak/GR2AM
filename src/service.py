import multiprocessing
import time
from multiprocessing import Queue
from gesture_capturing import GestureCapture
import json
import requests
from src.application.config import config

class Service(multiprocessing.Process):
    def __init__(self, dQueue: Queue):
        super().__init__()
        self.dQueue = dQueue

    def run(self):
        while True:
            if not self.dQueue.empty():
                self.triggerService(self.dQueue.get())

    def triggerService(self, data):# -> int:
        """Read dQueue and based on the result initiate the service"""
        print(GestureCapture.translate_class(data))
        gesture_name = GestureCapture.translate_class(data)

        with open("application/static/js/gesture_application_mapping.json", "r") as jsonFile:
            mappings = json.load(jsonFile)
            jsonFile.close()

        if "Negative" in gesture_name:
            # Keep Calm and Do Nothing
            pass
        else:
            if mappings[gesture_name]:
                app = mappings[gesture_name]
        # app = "Volume Up"
        app = app.strip().replace(' ', '_').lower()
        # Depending on app name, trigger the corresponding application
        print(app)
        try:
            function = getattr(self, app)
            function()
        except(AttributeError):
            #Do Nothing
            pass

    def brightness_up(self):
        base_url = config.CLIENT_URL + "brightness_change/up"
        requests.get(base_url)
        # time.sleep(1)

    def brightness_down(self):
        base_url = config.CLIENT_URL + "brightness_change/up"
        requests.get(base_url)
        # time.sleep(1)

    def volume_up(self):
        base_url = config.CLIENT_URL + "volume_change/up"
        requests.get(base_url)
        # time.sleep(1)

    def volume_down(self):
        base_url = config.CLIENT_URL + "volume_change/down"
        requests.get(base_url)
        # time.sleep(1)

    def toggle_play_pause(self):
        base_url = config.CLIENT_URL + "play_pause"
        requests.get(base_url)

    def media_next(self):
        base_url = config.CLIENT_URL + "media_change/next"
        requests.get(base_url)
        # time.sleep(1)

    def media_previous(self):
        base_url = config.CLIENT_URL + "media_change/prev"
        requests.get(base_url)
        # time.sleep(1)



