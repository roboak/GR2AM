import io
import json
import os
from pathlib import Path

import cv2
import mediapipe as mp
from src.utils.dataclass import GestureMetaData


class GestureCapture:
    def __init__(self, folder_location: str, gesture_meta_data: GestureMetaData, camera_input_value: int):
        self.gestureMetaData = gesture_meta_data
        self.folder_location = folder_location
        self.meta_data_file = str(Path(self.folder_location) / "MetaData.json")

        self.camera_input_value = camera_input_value
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

        if not (os.path.isfile(self.meta_data_file) and os.access(self.meta_data_file, os.R_OK)):
            with io.open(self.meta_data_file, 'w') as db_file:
                db_file.write(json.dumps({}))
        file = open(self.meta_data_file)
        self.gesture_dict = json.load(file)
        file.close()

        self.setup_cap()

    def setup_cap(self):
        if not (self.gestureMetaData.gestureName in self.gesture_dict.keys()):
            self.gesture_dict[self.gestureMetaData.gestureName] = self.gestureMetaData.__dict__
        self.gesture_name = self.gestureMetaData.gestureName + '_' + str(
            self.gesture_dict[self.gestureMetaData.gestureName]["trials"] + 1) + '.txt'

        print(self.gesture_name)

        self.gesture_path = self.folder_location + '/' + self.gesture_name

        self.all_keypoints = []

    #
    # def gesture_extraction(self, previous_frame, current_frame):
    #     current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    #     current_frame_gray = cv2.GaussianBlur(current_frame_gray, (21, 21), 0)
    #     previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    #     previous_frame_gray = cv2.GaussianBlur(previous_frame_gray, (21, 21), 0)
    #     frame_diff = cv2.absdiff(current_frame_gray, previous_frame_gray)
    #     cv2.imshow('frame diff ', frame_diff)

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

    def get_frame(self):
        cap = cv2.VideoCapture(self.camera_input_value)
        # Capture the video frame by frame
        # success, current_frame = cap.read()
        # image = None
        # while not success:
        #    success, current_frame = cap.read()
        # previous_frame = cv2.flip(current_frame, 1)

        record, redo = False, False
        end = False
        while cap.isOpened() and not end:

            # result stores the hand points extracted from mediapipe
            _, image = cap.read()
            image = cv2.flip(image, 1)  # mirror invert camera image

            if record:
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

            cv2.imshow('MediaPipe Hands', image)

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
                pass
            elif k & 0xFF == ord('r'):  # redo capture
                record = False
                self.setup_cap()
        # After the loop release the cap object
        cap.release()

        # Destroy all the windows
        cv2.destroyAllWindows()

    def write_file(self):
        if self.all_keypoints:  # only do smth when we have data to write
            self.data_file = open(self.gesture_path, "w")  # open file
            for item in self.all_keypoints:  # write all frames
                self.data_file.write(str(item) + "\n")
            self.data_file.close()  # close and save file

            self.update_meta_data(self.gesture_dict, self.gesture_name)  # update the meta file

    def update_meta_data(self, gesture_dictionary, gesture_file):
        gesture_dictionary[self.gestureMetaData.gestureName]["trials"] += 1
        gesture_dictionary[self.gestureMetaData.gestureName]["files"].append(gesture_file)
        with open(self.meta_data_file, "w") as outfile:
            json.dump(gesture_dictionary, outfile)
