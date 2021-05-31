import cv2


def GestureExtraction(previous_frame, current_frame):
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    current_frame_gray = cv2.GaussianBlur(current_frame_gray, (21, 21), 0)
    previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    previous_frame_gray = cv2.GaussianBlur(previous_frame_gray, (21, 21), 0)

    frame_diff = cv2.absdiff(current_frame_gray, previous_frame_gray)
    cv2.imshow('frame diff ', frame_diff)
