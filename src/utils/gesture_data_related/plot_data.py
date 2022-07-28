import ast
import re

import cv2
import numpy as np
import pandas as pd

lines = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9),
         (9, 13), (13, 17), (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20),
         (0, 17)]


def plot_landmarks(image, hand_landmarks, color):
    center_coord = return_scaled_hand_cordinates(image, hand_landmarks)
    cv2.circle(image, center_coord, 2, color, 2)


def draw_line(image, point1, point2):
    point1 = return_scaled_hand_cordinates(image, point1)
    point2 = return_scaled_hand_cordinates(image, point2)
    cv2.line(image, point1, point2, (255, 0, 0), 2)


def return_scaled_hand_cordinates(image, point):
    return int(point['X'] * image.shape[1]), int(point['Y'] * image.shape[0])


def plot_data(file_path, image_shape):
    with open(file_path, 'r') as file:
        # Get the correct label based on the file name
        label = re.search(r'gesture_._\w+_(\d+)_\d+\.txt', file_path).group(1)
        # Read all frames
        dataframes = file.readlines()
        for frame in dataframes:
            plot_image = np.zeros((image_shape[0], image_shape[0], 3), np.uint8)
            frame = ast.literal_eval(frame)
            df = pd.DataFrame(frame)
            for _, point in df.iterrows():
                plot_landmarks(plot_image, point, (0, 255, 0))
            point1 = {}
            point2 = {}
            for line_tuple in lines:
                point1['X'] = df.at[line_tuple[0], 'X']
                point1['Y'] = df.at[line_tuple[0], 'Y']
                point2['X'] = df.at[line_tuple[1], 'X']
                point2['Y'] = df.at[line_tuple[1], 'Y']
                draw_line(plot_image, point1, point2)

            cv2.imshow('Plot', plot_image)

            if cv2.waitKey(0) & 0xFF == ord('q'):  # close on key q
                break
    cv2.destroyAllWindows()


plot_data("/home/akash/Documents/HLCV_Project/GR2AM/HandDataset/train/u1_gesture_i_tap_4_1.txt", (720, 1280))
