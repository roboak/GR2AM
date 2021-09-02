import io
import json
import os
from multiprocessing import Queue
from pathlib import Path
from typing import NamedTuple
import cv2
import mediapipe as mp
from pynput import keyboard

from src.utils.gesture_data_related.dataclass import GestureMetaData





class GestureCapture:
    def __init__(self, camera_input_value: int, folder_location: str = "", gesture_meta_data: GestureMetaData = None,
                 aQueue: Queue = None, cQueue: Queue = None, dQueue: Queue = None, window_size=30, frontend=False):

        self.gesture_name = None
        self.gesture_path = None
        self.key_capture = None
        self.all_key_points = []
        self.live_frame_size = window_size

        self.camera_input_value = camera_input_value
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

        if gesture_meta_data is not None:
            """
            The metadata for the gestures.
            Meaning which ones were created and how many iterations do they have an so on.
            """
            self.gestureMetaData = gesture_meta_data
            self.folder_location = folder_location
            self.meta_data_file = str(Path(self.folder_location) / "MetaData.json")

            if not (os.path.isfile(self.meta_data_file) and os.access(self.meta_data_file, os.R_OK)):
                with io.open(self.meta_data_file, 'w') as db_file:
                    db_file.write(json.dumps({}))
            with open(self.meta_data_file) as file:
                self.gesture_dict = json.load(file)
            self.live = False
            self.preventRecord = False
        else:
            self.live = True
            self.preventRecord = True

        self.aQueue = aQueue
        self.cQueue = cQueue
        self.dQueue = dQueue

        # TODO: those two lines needed for flask but will stop normal UI from starting
        if frontend:
            self.keyboard_listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
            self.keyboard_listener.start()

    def on_press(self, key):
        self.key_capture = key
        if key == keyboard.Key.esc:
            # Stop listener
            return False

    def on_release(self, key):
        # self.key_capture = None
        pass

    def setup_cap(self):
        if not (self.gestureMetaData.gestureName in self.gesture_dict.keys()):
            self.gesture_dict[self.gestureMetaData.gestureName] = self.gestureMetaData.__dict__
        self.gesture_name = self.gestureMetaData.gestureName + '_' + str(
            self.gesture_dict[self.gestureMetaData.gestureName]["trials"] + 1) + '.txt'
        self.gesture_path = self.folder_location + '/' + self.gesture_name

        print(self.gesture_name)

    def get_frame(self):
        # Setup for right file to be recorded
        if not self.live:
            self.setup_cap()

        cap = cv2.VideoCapture(self.camera_input_value)
        last_result = ""

        record, redo, end = False, False, False
        while cap.isOpened() and not end:
            # result stores the hand points extracted from MediaPipe
            _, image = cap.read()
            image = cv2.flip(image, 1)  # mirror invert camera image

            # Record frames to all_key_points
            if record or self.live:
                self.record_frame(image)

            # Display mark to indicate file over length
            if record and len(self.all_key_points) >= self.live_frame_size:
                cv2.putText(image, "!", (150, 100), cv2.QT_FONT_NORMAL, 2, (0, 0, 255, 255), 2)

            # Collect results from classifying process
            if self.cQueue and not self.cQueue.empty():
                last_result = str(self.cQueue.get())

            # The collected result will be pushed inside dQueue which will be consumed by Service process to trigger
            # applications
            if last_result and self.dQueue:
                self.dQueue.put(last_result)

            if last_result:  # If a result is present display it
                cv2.putText(image, "Last class: " + self.translate_class(last_result), (10, 50), cv2.QT_FONT_NORMAL, 1,
                            (0, 0, 255, 255), 2)  # BGR of course

            cv2.imshow('MediaPipe Hands', image)

            # Keyboard bindings
            k = cv2.waitKey(1)  # read key pressed event
            try:
                if k == '-1':
                    pass  # no key press (don't waste time)
                elif k % 256 == 32:  # spacebar to record
                    if not self.preventRecord:
                        record = not record
                        print("Toggle Recording Mode")
                    else:
                        self.live = True
                elif k & 0xFF == ord('q'):  # close on key q
                    record = False
                    self.write_file()
                    end = True
                elif k & 0xFF == ord('n'):  # next capture
                    record = False
                    self.write_file()
                    self.setup_cap()
                elif k & 0xFF == ord('r'):  # redo capture
                    record = False
                    self.all_key_points = []
            except AttributeError as e:
                print("Gesture Capturing Attribute Error: ", AttributeError, e)
                pass  # TODO figure this one out

        # After the loop release the cap object
        cap.release()

        # Destroy all the windows
        cv2.destroyAllWindows()

    def get_frame_yield(self):
        # Setup for right file to be recorded
        if not self.live:
            self.setup_cap()

        cap = cv2.VideoCapture(self.camera_input_value)
        last_result = ""

        record, redo, end = False, False, False
        while cap.isOpened() and not end:
            # result stores the hand points extracted from MediaPipe
            _, image = cap.read()
            image = cv2.flip(image, 1)  # mirror invert camera image

            # Record frames to all_key_points
            if record or self.live:
                self.record_frame(image)

            # Display mark to indicate file over length
            if record and len(self.all_key_points) >= self.live_frame_size:
                cv2.putText(image, "!", (150, 100), cv2.QT_FONT_NORMAL, 2, (0, 0, 255, 255), 2)

            # Collect results from classifying process
            if self.cQueue and not self.cQueue.empty():
                last_result = str(self.cQueue.get())

            # The collected result will be pushed inside dQueue which will be consumed by Service process to trigger
            # applications
            if last_result and self.dQueue:
                self.dQueue.put(last_result)

            if last_result:  # If a result is present display it
                cv2.putText(image, "Last class: " + self.translate_class(last_result), (10, 50), cv2.QT_FONT_NORMAL, 1,
                            (0, 0, 255, 255), 2)  # BGR of course

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

            # Keyboard bindings
            # k = cv2.waitKey(1)  # read key pressed event
            try:
                if not self.key_capture:
                    pass  # Don't waste my time, printing errorsr
                elif self.key_capture.char == 's':  # s to record
                    if not self.preventRecord:
                        record = not record
                        print("Toggle Recording Mode")
                    else:
                        self.live = True
                elif self.key_capture.char == 'n':  # next capture
                    record = False
                    self.write_file()
                    self.setup_cap()
                elif self.key_capture.char == 'r':  # redo capture
                    record = False
                    self.all_key_points = []
            except AttributeError as e:
                print("Gesture Capturing Attribute Error: ", AttributeError, e)
                pass  # TODO figure this one out
            finally:
                self.key_capture = None

        # After the loop release the cap object
        cap.release()

        # Destroy all the windows
        cv2.destroyAllWindows()

    def get_hand_points(self, image) -> NamedTuple("res", [('multi_hand_landmarks', list), ('multi_handedness', list)]):
        with self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.7) as hands:
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
        key_points_per_frame = []
        if results.multi_hand_landmarks:
            for data_point in results.multi_hand_landmarks[0].landmark:
                if data_point:
                    key_points_per_frame.append({
                        'X': data_point.x,
                        'Y': data_point.y,
                        'Z': data_point.z,
                    })
        if key_points_per_frame:
            if self.aQueue:
                self.aQueue.put(key_points_per_frame)
            else:
                self.all_key_points.append(key_points_per_frame)

    def write_file(self):
        if self.all_key_points and not self.live:  # only do something when we have data to write
            with open(self.gesture_path, "w") as data_file:  # open file and save/close afterwards
                for item in self.all_key_points:  # write all frames
                    data_file.write(str(item) + "\n")

            self.update_meta_data(self.gesture_dict, self.gesture_name)  # update the meta file
            self.all_key_points = []

    def update_meta_data(self, gesture_dictionary, gesture_file):
        gesture_dictionary[self.gestureMetaData.gestureName]["trials"] += 1
        gesture_dictionary[self.gestureMetaData.gestureName]["files"].append(gesture_file)
        with open(self.meta_data_file, "w") as outfile:
            json.dump(gesture_dictionary, outfile)

    @staticmethod
    def translate_class(classification_id: str) -> str:
        if not classification_id.isdigit():
            print("wats dat?!" + classification_id)
            return ''
        classification_id = int(classification_id)

        classes = {0: 'Thumb tap', 1: 'Thumb Swipe Up', 2: 'Thumb Swipe Down',
                   3: 'Index tap', 4: 'Index Swipe Up', 5: 'Index Swipe Down',
                   6: 'Middle tap', 7: 'Middle Swipe Up', 8: 'Middle Swipe Down',
                   9: 'Ring tap', 10: 'Ring Swipe Up', 11: 'Ring Swipe Down',
                   12: 'Little tap', 13: 'Little Swipe Up', 14: 'Little Swipe Down',
                   15: 'Negative_still', 16: 'Negative_up', 17: 'Negative_down'}

        return classes[classification_id]
