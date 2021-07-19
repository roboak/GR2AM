import math
import time
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
import numpy as np
import GetHandPoints
from scipy.spatial import distance
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
base_scale = 0.08

def normalize_scale(raw_points):
    distance_5_17 = find_distance(raw_points, (5, 17))
    print("distance: ", distance_5_17)
    scale_factor = base_scale / distance_5_17
    hand_data = raw_points.landmark
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


def plot_landmarks(image, hand_landmarks, color):
    for point in hand_landmarks:
        center_coord = return_scaled_hand_cordinates(image, point)
        cv2.circle(image, center_coord, 2, color, 2)


def return_scaled_hand_cordinates(image, point):
    return int(point.x*image.shape[1]), int(point.y*image.shape[0])


def GetFrame():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)
    pTime = 0
    while cap.isOpened():
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        print("fps: ", fps)
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        image = cv2.flip(image, 1)
        plot_image = np.zeros(image.shape, np.uint8)
        results = GetHandPoints.GetHandPoints(image)
        try:
            if results.multi_handedness[0].classification[0].label == 'Right' and \
                    results.multi_handedness[0].classification[0].score>0.60:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    mp_drawing.draw_landmarks(plot_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    plot_landmarks(plot_image, hand_landmarks.landmark, (255, 255, 255))
                normalized_hand_data = normalize_scale(results.multi_hand_landmarks[0])
                filter(results.multi_hand_landmarks[0])
                plot_landmarks(plot_image, normalized_hand_data, (255, 120, 100))
        except(TypeError):
            pass
        cv2.imshow('MediaPipe Hands', image)
        cv2.imshow('Plot', plot_image)

        if cv2.waitKey(2) & 0xFF == ord('q'):  # close on key q
            plt.plot(ratio_plt)
            plt.show()

            break

    cap.release()
    cv2.destroyAllWindows()

def find_distance(raw_points, points):
    point_1 = [raw_points.landmark[points[0]].x, raw_points.landmark[points[0]].y, raw_points.landmark[points[0]].z]
    point_2 = [raw_points.landmark[points[1]].x, raw_points.landmark[points[1]].y, raw_points.landmark[points[1]].z]
    distance_1_2 = distance.euclidean(point_1, point_2)
    return distance_1_2


min_r = 1000
max_r = 0
ratio_plt = []
def filter(raw_points):

    # Shape of hand based on dinstances
    d5_17 = find_distance(raw_points, (5, 17))
    d0_5 = find_distance(raw_points, (0, 5))
    d0_17 = find_distance(raw_points, (0, 17))


    print("distance d5_17/d0_5 = {}".format(d5_17/d0_5))
    print("distance d0_5/d0_17 = {}".format(d0_5/d0_17))
    print("distance d0_17/d5_17 = {}".format(d0_17/d5_17))
    print("mean ", (d5_17/d0_5 + d0_5/d0_17 +d0_17/d5_17 )/3)
    ratio_plt.append(d0_17/d5_17)


GetFrame()
