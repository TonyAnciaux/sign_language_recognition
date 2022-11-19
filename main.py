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
    #image.flags.writable = False
    results = holistic_model.process(image)
    #image.flags.writable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,\
        mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),\
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,\
        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),\
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,\
        mp_drawing.DrawingSpec(color=(80, 22, 76), thickness=2, circle_radius=4),\
            mp_drawing.DrawingSpec(color=(80, 44, 250), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,\
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),\
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))



# accessing webcam, looping through the frames in video feed and displaying them 
cap = cv2.VideoCapture(0)
#  Set Mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic_model:
    while cap.isOpened():
        ret, frame = cap.read()
        image, results = mediapipe_detection(frame, holistic_model)
        draw_landmarks(image, results)
        cv2.imshow("Webcam Feed", image)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
cap.release()
cv2.destroyAllWindows()



