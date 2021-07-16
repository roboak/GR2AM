import math

import cv2
import mediapipe as mp
import numpy as np
import GetHandPoints
from scipy.spatial import distance
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
base_scale = 0.08


def Normalise_Size(hand_data, scale):
    for value in hand_data:
        value.x = value.x * scale
        value.y = value.y * scale
        value.z = value.z * scale
    return hand_data


def normalize_scale(image, raw_points):
    point_5 = return_scaled_hand_cordinates_1(image, raw_points.landmark[5])
    point_17 = return_scaled_hand_cordinates_1(image, raw_points.landmark[17])
    distance_5_17 = distance.euclidean([point_5[0], point_5[1]], [point_17[0], point_17[1]])
    print("distance: ", distance_5_17)
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
    counter = 0
    sum_x = 0
    sum_y = 0
    sum_z = 0
    for value in hand_data:
        sum_x += value.x
        sum_y += value.y
        sum_z += value.z
        counter += 1
    print(counter)
    mean_x = sum_x / counter
    mean_y = sum_y / counter
    mean_z = sum_z / counter
    for value in hand_data:
        value.x -= mean_x
        value.y -= mean_y
        value.z -= mean_z
    """
    reference_x = hand_data[0].x - 0.5
    reference_y = hand_data[0].y - 0.5
    reference_z = hand_data[0].z - 0.5
    for value in hand_data:
        value.x = value.x - reference_x
        value.y = value.y - reference_y
        value.z = value.z - reference_z"""

    return hand_data


def plot_landmarks(image, hand_landmarks, color):
    for point in hand_landmarks:
        center_coord = return_scaled_hand_cordinates(image, point)
        cv2.circle(image, center_coord, 10, color, 2)


def return_scaled_hand_cordinates(image, point):
    return int(point.x*image.shape[1]), int(point.y*image.shape[0])


def return_scaled_hand_cordinates_1(image, point):
    return point.x*1, point.y*1


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
            normalized_hand_data = normalize_scale(image, results.multi_hand_landmarks[0])
            # pt_5 = return_scaled_hand_cordinates(image, results.multi_hand_landmarks[0].landmark[5])
            # pt_17 = return_scaled_hand_cordinates(image, results.multi_hand_landmarks[0].landmark[17])
            # distance_5_17 = math.dist([pt_5[0], pt_5[1]], [pt_17[0], pt_17[1]])
            # distance_5_17 = distance.euclidean([pt_5[0], pt_5[1]], [pt_17[0], pt_17[1]])
            # scale_factor = base_scale/distance_5_17
            # normalised_hand_landmarks_list = Normalise_Size(results.multi_hand_landmarks[0].landmark, scale_factor)
            # pt_5_norm = return_scaled_hand_cordinates(image, normalized_hand_data[5])
            # pt_17_norm = return_scaled_hand_cordinates(image, normalized_hand_data[17])
            # normalized_length = distance.euclidean([pt_5_norm[0], pt_5_norm[1]], [pt_17_norm[0], pt_17_norm[1]])
            plot_landmarks(plot_image, normalized_hand_data, (255, 120, 100))
            # print("length: ", distance_5_17)
            # print(f"length normalized: {normalized_length}")

        cv2.imshow('MediaPipe Hands', image)
        cv2.imshow('Plot', plot_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # close on key q
            break

    cap.release()
    cv2.destroyAllWindows()


GetFrame()
