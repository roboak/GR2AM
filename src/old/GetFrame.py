import cv2
import mediapipe as mp

import GetHandPoints

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def GetFrame():

    file = open("../HandDataset/gesture3.txt", "w")

    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(r"/Users/jsonnet/OneDrive/Studium/PyCom/project/Test_gesture.mp4")
    # Capture the video frame by frame
    success, current_frame = cap.read()
    while (not success):
        success, current_frame = cap.read()
    previous_frame = cv2.flip(current_frame, 1)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        # result stores the handpoints extracted from mediapipe
        image = cv2.flip(image, 1)
        ## Get hand joints using MediaPipe
        results = GetHandPoints.GetHandPoints(image)
        # Display the resulting frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # print(results.multi_hand_landmarks)  # one frame of 21 landmarks
        # print(dir(results.multi_hand_landmarks[0]))

        keypoints = []
        if results.multi_hand_landmarks:
            for data_point in results.multi_hand_landmarks[0].landmark:
                keypoints.append({
                    'X': data_point.x,
                    'Y': data_point.y,
                    'Z': data_point.z,
                })

        print(keypoints)

        file.write(str(keypoints) + "\n")


        cv2.imshow('MediaPipe Hands', image)

        # Frame Differencing applied to extract moving objects
        # current_frame = image.copy()
        # GestureExtraction.GestureExtraction(previous_frame, current_frame)
        # previous_frame = current_frame.copy()

        if cv2.waitKey(2) & 0xFF == ord('q'):  # close on key q
            break


    file.close()


    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
