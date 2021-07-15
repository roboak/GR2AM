import copy
import io
import json
import os
import time
from pathlib import Path
from multiprocessing import Queue

import cv2
import mediapipe as mp
from scipy.spatial import distance

from src.utils.dataclass import GestureMetaData


base_scale = 65


def normalize_scale(image, raw_points):
    point_5 = return_scaled_hand_cordinates(image, raw_points.landmark[5])
    point_17 = return_scaled_hand_cordinates(image, raw_points.landmark[17])
    distance_5_17 = distance.euclidean([point_5[0], point_5[1]], [point_17[0], point_17[1]])
    scale_factor = base_scale / distance_5_17
    hand_data = raw_points.landmark
    hand_data_2 = [{"x": value.x * scale_factor, "y": value.y * scale_factor, "z": value.z * scale_factor}
                   for value in hand_data]
    reference_x = hand_data[0].x   # - 0.5  # (image.shape[1] / 2)
    reference_y = hand_data[0].y   # - 0.5  # (image.shape[0] / 2)
    for value in hand_data:
        value.x = value.x * scale_factor
        value.y = value.y * scale_factor
        value.z = value.z * scale_factor
    # TODO: Does it make sense here? I thought to normalize wrt the wrist then translate the whole hand to the middle
        value.x = value.x - reference_x
        value.y = value.y - reference_y
    reference_x = hand_data[0].x - 0.5
    reference_y = hand_data[0].y - 0.5
    reference_z = hand_data[0].z - 0.5
    for value in hand_data:
        value.x = value.x - reference_x
        value.y = value.y - reference_y
        value.z = value.z - reference_z

    return hand_data


def return_scaled_hand_cordinates(image, point):
    return int(point.x*image.shape[1]), int(point.y*image.shape[0])


class GestureCapture:
    def __init__(self, camera_input_value: int, folder_location: str = "", gesture_meta_data: GestureMetaData = None,
                 aQueue: Queue = None, bQueue: Queue = None, window_size=40):
        self.gesture_name = None
        self.gesture_path = None
        self.all_keypoints = []
        self.last_append = 0
        self.live_framesize = window_size

        self.camera_input_value = camera_input_value
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

        if gesture_meta_data is not None:
            self.gestureMetaData = gesture_meta_data
            self.folder_location = folder_location
            self.meta_data_file = str(Path(self.folder_location) / "MetaData.json")

            if not (os.path.isfile(self.meta_data_file) and os.access(self.meta_data_file, os.R_OK)):
                with io.open(self.meta_data_file, 'w') as db_file:
                    db_file.write(json.dumps({}))
            with open(self.meta_data_file) as file:
                self.gesture_dict = json.load(file)
            self.live = False
        else:
            self.live = True

        self.aQueue = aQueue
        self.bQueue = bQueue

    def setup_cap(self):
        if not (self.gestureMetaData.gestureName in self.gesture_dict.keys()):
            self.gesture_dict[self.gestureMetaData.gestureName] = self.gestureMetaData.__dict__
        self.gesture_name = self.gestureMetaData.gestureName + '_' + str(
            self.gesture_dict[self.gestureMetaData.gestureName]["trials"] + 1) + '.txt'
        self.gesture_path = self.folder_location + '/' + self.gesture_name

        print(self.gesture_name)

    def get_frame(self):
        if not self.live:
            self.setup_cap()

        cap = cv2.VideoCapture(self.camera_input_value)
        # cap = cv2.VideoCapture('raw_recording.mov')
        last_result = ""

        record, redo = False, False
        end = False
        while cap.isOpened() and not end:

            # result stores the hand points extracted from mediapipe
            _, image = cap.read()
            image = cv2.flip(image, 1)  # mirror invert camera image

            # Record frames to all_keypoints
            if record or self.live:
                self.record_frame(image)

            if self.live and self.all_keypoints:

                # When 60 frames are captured create job to classify
                if len(self.all_keypoints) == self.live_framesize:
                    self.aQueue.put(copy.copy(self.all_keypoints))

                    # Record overlapping window
                    self.all_keypoints = self.all_keypoints[:-self.live_framesize//2]  # save last 20 entries for next window

                    cv2.putText(image, ".", (150, 100), cv2.QT_FONT_NORMAL, 1, (0, 255, 0, 255), 2)

                # When 10s from the last frame have passed create job (cond. have at least 21 frames due to overlap)
                if len(self.all_keypoints) > 20 and time.time() >= self.last_append + 10:
                    self.aQueue.put(copy.copy(self.all_keypoints))

                    # empty out completely, no related movements
                    self.all_keypoints = []

                # Collect results
                if not self.bQueue.empty():
                    last_result = str(self.bQueue.get())

            if self.live and last_result:  # In live mode always display text
                cv2.putText(image, "Last class: " + self.translate_class(last_result), (10, 50), cv2.QT_FONT_NORMAL, 1,
                            (0, 0, 255, 255), 2)  # BGR of course

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

        # After the loop release the cap object
        cap.release()

        # Destroy all the windows
        cv2.destroyAllWindows()

    def get_hand_points(self, image):
        with self.mp_hands.Hands(min_detection_confidence=0.4, min_tracking_confidence=0.4) as hands:
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

            landmark = results.multi_hand_landmarks[0].landmark

            if self.live:
                landmark = normalize_scale(image, results.multi_hand_landmarks[0])

            for data_point in landmark:
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
        if self.all_keypoints and not self.live:  # only do smth when we have data to write
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

    @staticmethod
    def translate_class(classid: str) -> str:
        if not classid.isdigit():
            print("wats dat?!" + classid)
            return ''
        classid = int(classid)

        classes = {0: 'Thumb tap', 1: 'Thumb Swipe Up', 2: 'Thumb Swipe Down',
                   3: 'Index tap', 4: 'Index Swipe Up', 5: 'Index Swipe Down',
                   6: 'Middle tap', 7: 'Middle Swipe Up', 8: 'Middle Swipe Down',
                   9: 'Ring tap', 10: 'Ring Swipe Up', 11: 'Ring Swipe Down',
                   12: 'Little tap', 13: 'Little Swipe Up', 14: 'Little Swipe Down',
                   15: 'Negative'}

        return classes[classid]
