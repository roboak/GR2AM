import math

import cv2
import mediapipe as mp
import numpy as np
import GetHandPoints

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
base_scale = 65

def Normalise_Size(hand_data, scale):
    for value in hand_data:
        temp_dict = {}
        value.x = value.x*scale
        value.y = value.y * scale
        value.z = value.z * scale
    return(hand_data)



def plot_landmarks(image, hand_landmarks, color):
    for point in hand_landmarks:
        center_coord = return_scaled_hand_cordinates(image, point)
        cv2.circle(image, center_coord, 10, color, 2)
def return_scaled_hand_cordinates(image, point):
    return (int(point.x*image.shape[1]), int(point.y*image.shape[0]))


def GetFrame():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        image = cv2.flip(image, 1)
        plot_image = np.zeros(image.shape, np.uint8)
        results = GetHandPoints.GetHandPoints(image)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(plot_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                plot_landmarks(plot_image, hand_landmarks.landmark, (255, 255, 255))

            pt_5 = return_scaled_hand_cordinates(image, results.multi_hand_landmarks[0].landmark[5])
            pt_17 = return_scaled_hand_cordinates(image, results.multi_hand_landmarks[0].landmark[17])
            distance_5_17 = math.dist([pt_5[0], pt_5[1]], [pt_17[0], pt_17[1]])
            scale_factor = base_scale/distance_5_17
            normalised_hand_landmarks_list = Normalise_Size(results.multi_hand_landmarks[0].landmark, scale_factor)
            print ("Wrist: ", normalised_hand_landmarks_list[0])
            print ("Middle Finger Top: ", normalised_hand_landmarks_list[12])

            plot_landmarks(plot_image, normalised_hand_landmarks_list, (255, 120, 100))
            print("length: ", distance_5_17)

        cv2.imshow('MediaPipe Hands', image)
        cv2.imshow('Plot', plot_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # close on key q
            break

    cap.release()
    cv2.destroyAllWindows()


GetFrame()