import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import mediapipe as mp


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def mediapipe_detection(image, holistic_model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writable = False
    results = holistic_model.process(image)
    image.flags.writable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


# accessing webcam, looping through the frames in video feed and displaying them 
cap = cv2.VideoCapture(0)
#  Set Mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic_model:
    while cap.isOpened():
        ret, frame = cap.read()
        image, results = mediapipe_detection(frame, holistic_model)
        cv2.imshow("Webcam Feed", frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
cap.release()
cv2.destroyAllWindows()
