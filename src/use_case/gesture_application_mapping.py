import multiprocessing
from multiprocessing import Queue
from src.use_case.gesture_capturing import GestureCapture
import json
import requests
from src.frontend.config import config
from flask import session


class ApplicationTriggeringService(multiprocessing.Process):
    def __init__(self, dQueue: Queue, username: str = ''):
        super().__init__()
        self.dQueue = dQueue
        self.username = username

    def run(self):
        while True:
            if not self.dQueue.empty():
                self.trigger_service(self.dQueue.get_nowait())

    def trigger_service(self, data):
        """Read dQueue and based on the result initiate the service"""
        # print("data:", data)
        gesture_id = GestureCapture.translate_class_to_gesture_id(data)
        # print("Gesture Class name: ", gesture_id)

        path_mapping = "../frontend/static/js/" + self.username + "/gesture_application_mapping.json"
        # print("path_mapping: ", path_mapping)
        with open(path_mapping, "r") as jsonFile:
            mappings = json.load(jsonFile)
            jsonFile.close()

        if "Negative" in gesture_id:
            # Keep Calm and Do Nothing
            pass
        else:
            if gesture_id in mappings.keys():
                app = list(mappings[gesture_id][1].keys())[0]
                print(app)
                try:
                    function = getattr(self, app)
                    function()
                except AttributeError:
                    # Do Nothing
                    pass

    def brightness_up(self):
        base_url = config.CLIENT_URL + "brightness_change/up"
        requests.get(base_url)

    def brightness_down(self):
        base_url = config.CLIENT_URL + "brightness_change/up"
        requests.get(base_url)

    def volume_up(self):
        base_url = config.CLIENT_URL + "volume_change/up"
        requests.get(base_url)

    def volume_down(self):
        base_url = config.CLIENT_URL + "volume_change/down"
        requests.get(base_url)

    def toggle_play_pause(self):
        base_url = config.CLIENT_URL + "play_pause"
        requests.get(base_url)

    def media_next(self):
        base_url = config.CLIENT_URL + "media_change/next"
        requests.get(base_url)

    def media_previous(self):
        base_url = config.CLIENT_URL + "media_change/prev"
        requests.get(base_url)


if __name__ == '__main__':
    dQueue = Queue()
    ams = ApplicationTriggeringService(dQueue, 'test_user')
    ams.start()
    i = 0
    while (i < 2):
        i += 1
        dQueue.put('1')
