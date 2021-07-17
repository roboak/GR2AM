import cv2
import numpy as np
import re
import pandas as pd
import ast

def plot_landmarks(image, hand_landmarks, color):
    center_coord = return_scaled_hand_cordinates(image, hand_landmarks)
    cv2.circle(image, center_coord, 2, color, 2)


def return_scaled_hand_cordinates(image, point):
    return int(point['X']*image.shape[1]), int(point['Y']*image.shape[0])



def plot_data(file_path, image_shape):
    with open(file_path, 'r') as file:
        # Get the correct label based on the file name
        label = re.search(r'gesture_._\w+_(\d+)_\d+\.txt', file_path).group(1)
        # Read all frames
        dataframes = file.readlines()
        for frame in dataframes:
            plot_image = np.zeros(image_shape, np.uint8)
            frame = ast.literal_eval(frame)
            df = pd.DataFrame(frame)
            for _, point in df.iterrows():
                plot_landmarks(plot_image, point, (255, 0, 0))
            cv2.imshow('Plot', plot_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # close on key q
                break
    cv2.destroyAllWindows()


plot_data("D:\Work\LumosNox\LumosNox\HandDataset\Josh\gesture_i_down_6_1.txt", (480,640))

