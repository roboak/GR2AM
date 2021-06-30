import io
import json
import os
import threading
from pathlib import Path

import cv2
import mediapipe as mp
from src.utils.dataclass import GestureMetaData

from concurrent.futures import ThreadPoolExecutor, as_completed
import time


class GestureCapture:
    def __init__(self, folder_location: str, gesture_meta_data: GestureMetaData, camera_input_value: int):
        self.gesture_name = None
        self.gesture_path = None
        self.all_keypoints = []
        self.last_append = 0
        
        self.gestureMetaData = gesture_meta_data
        self.folder_location = folder_location
        self.meta_data_file = str(Path(self.folder_location) / "MetaData.json")

        self.camera_input_value = camera_input_value
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

        if not (os.path.isfile(self.meta_data_file) and os.access(self.meta_data_file, os.R_OK)):
            with io.open(self.meta_data_file, 'w') as db_file:
                db_file.write(json.dumps({}))
        with open(self.meta_data_file) as file:
            self.gesture_dict = json.load(file)

        self.executor = ThreadPoolExecutor(max_workers=4)
        self.futures = []
        # TODO to extract the results:
        # for future in as_completed(self.futures):
        #    print(future.result())

    def setup_cap(self):
        if not (self.gestureMetaData.gestureName in self.gesture_dict.keys()):
            self.gesture_dict[self.gestureMetaData.gestureName] = self.gestureMetaData.__dict__
        self.gesture_name = self.gestureMetaData.gestureName + '_' + str(
            self.gesture_dict[self.gestureMetaData.gestureName]["trials"] + 1) + '.txt'
        self.gesture_path = self.folder_location + '/' + self.gesture_name

        print(self.gesture_name)

    def classify_capture(self, frames):
        #print(frames[0])
        # print(threading.get_ident())
        # time.sleep(7)
        return 1

    def get_frame(self):
        self.setup_cap()

        cap = cv2.VideoCapture(self.camera_input_value)

        record, live, redo = False, False, False
        end = False
        while cap.isOpened() and not end:

            # result stores the hand points extracted from mediapipe
            _, image = cap.read()
            image = cv2.flip(image, 1)  # mirror invert camera image

            if record:
                self.record_frame(image)

            if live and self.all_keypoints:

                if len(self.all_keypoints) == 60:  # if 60 frames have been captured

                    # TODO check if a copy of the list is needed ? due to datarace
                    self.futures.append(self.executor.submit(self.classify_capture, frames=self.all_keypoints))

                    self.all_keypoints = self.all_keypoints[:-20]  # save last 20 entries for next round

                if len(self.all_keypoints) > 20 and time.time() >= self.last_append + 10:  # check for time but with at least 21 entries!
                    print("There was no new capture but a timeout of 10 secs")
                    # TODO
                    self.all_keypoints = []

            cv2.imshow('MediaPipe Hands', image)

            ## Keyboard bindings ##
            k = cv2.waitKey(1)  # read key pressed event
            if k % 256 == 32:  # spacebar to record
                record = not record
                print("Toggle Recording Mode")
            if k & 0xFF == ord('q'):  # close on key q
                record = False
                self.write_file()
                end = True
            elif k & 0xFF == ord('n'):  # next capture
                record = False
                self.write_file()
                self.setup_cap()
            elif k & 0xFF == ord('r'):  # redo capture
                record = False
                self.all_keypoints = []
                # self.setup_cap()
            elif k & 0xFF == ord('l'):  # live capture mode
                record = True
                live = True

        # After the loop release the cap object
        cap.release()

        # Destroy all the windows
        cv2.destroyAllWindows()

    def get_hand_points(self, image):
        with self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            # Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to pass by reference.
            image.flags.writeable = False
            results = hands.process(image)
            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return results

    def record_frame(self, image):
        # Get hand joints using MediaPipe
        results = self.get_hand_points(image)
        # Display the resulting frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        keypoints_per_frame = []
        if results.multi_hand_landmarks:
            for data_point in results.multi_hand_landmarks[0].landmark:
                if data_point:
                    keypoints_per_frame.append({
                        'X': data_point.x,
                        'Y': data_point.y,
                        'Z': data_point.z,
                    })
        if keypoints_per_frame:
            self.all_keypoints.append(keypoints_per_frame)
            self.last_append = time.time()

    def write_file(self):
        if self.all_keypoints:  # only do smth when we have data to write
            with open(self.gesture_path, "w") as data_file:  # open file and save/close afterwards
                for item in self.all_keypoints:  # write all frames
                    data_file.write(str(item) + "\n")

            self.update_meta_data(self.gesture_dict, self.gesture_name)  # update the meta file
            self.all_keypoints = []

    def update_meta_data(self, gesture_dictionary, gesture_file):
        gesture_dictionary[self.gestureMetaData.gestureName]["trials"] += 1
        gesture_dictionary[self.gestureMetaData.gestureName]["files"].append(gesture_file)
        with open(self.meta_data_file, "w") as outfile:
            json.dump(gesture_dictionary, outfile)
