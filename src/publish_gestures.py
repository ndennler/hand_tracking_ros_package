#/usr/bin/python3

#----------------------------------------------------
#
#       parses input arguments
#       The only requirement is the webcam id
#
#----------------------------------------------------
import os
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("webcam_id", type=int,
                    help="The number of the webcam device to open. It should be an integer (0 is the laptop's built-in webcam).")

args = parser.parse_args()

import sys
import rospy

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'

if ros_path in sys.path:
    sys.path.remove(ros_path)

import cv2

sys.path.append(ros_path)



import numpy as np
from numpy.lib.function_base import angle
import mediapipe as mp

from std_msgs.msg import Float64, String
import json 
from google.protobuf.json_format import MessageToDict
from util import landmark_to_array, get_angles_from_rotation


'''
Initialize the ROS node and set up some 
'''
rospy.init_node('thumb_node')



mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

gesture_publisher = rospy.Publisher('gesture', String, queue_size=10)

#these are the points used in estimating the rotation matrix of the hand
points_of_interest = [
            mp_hands.HandLandmark.INDEX_FINGER_MCP,
            mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
            mp_hands.HandLandmark.RING_FINGER_MCP,
            mp_hands.HandLandmark.PINKY_MCP,
        ]
        
if os.path.isfile('calib.npy'):
    print('file found!')
    unrotated_hand = 'Right' #stores the handedness
    unrotated = np.load('calib.npy')  
    print(unrotated)
else:    
    unrotated_hand = 'None' #stores the handedness
    unrotated = np.array([[-0.60397809, -0.10909924, -0.78949846],
     [-0.39785473, -0.16591713, -0.90232096],
     [-0.16918771, -0.15877018, -0.97271144],
     [ 0.03837452, -0.09477445, -0.99475887]]) #stores the "0" position

last_reading = np.nan #previous reading of thumb angle
decay = 0.5 #parameter for exponential smoohing filter (weights total of all past observatoins with new observation)
r = rospy.Rate(50) # 50hz






def simpleGesture(handLandmarks, handedness):

    thumbIsOpen = False
    indexIsOpen = False
    middelIsOpen = False
    ringIsOpen = False
    pinkyIsOpen = False

    pseudoFixKeyPoint = handLandmarks[2].x
    if  handedness=='Right' and handLandmarks[3].x < pseudoFixKeyPoint and handLandmarks[4].x < pseudoFixKeyPoint:
        thumbIsOpen = True

    if  handedness=='Left' and handLandmarks[3].x > pseudoFixKeyPoint and handLandmarks[4].x > pseudoFixKeyPoint:
        thumbIsOpen = True

    pseudoFixKeyPoint = handLandmarks[6].y
    if handLandmarks[7].y < pseudoFixKeyPoint and handLandmarks[8].y < pseudoFixKeyPoint:
        indexIsOpen = True

    pseudoFixKeyPoint = handLandmarks[10].y
    if handLandmarks[11].y < pseudoFixKeyPoint and handLandmarks[12].y < pseudoFixKeyPoint:
        middelIsOpen = True

    pseudoFixKeyPoint = handLandmarks[14].y
    if handLandmarks[15].y < pseudoFixKeyPoint and handLandmarks[16].y < pseudoFixKeyPoint:
        ringIsOpen = True

    pseudoFixKeyPoint = handLandmarks[18].y
    if handLandmarks[19].y < pseudoFixKeyPoint and handLandmarks[20].y < pseudoFixKeyPoint:
        pinkyIsOpen = True

    if thumbIsOpen and indexIsOpen and middelIsOpen and ringIsOpen and pinkyIsOpen:
        return "FIVE!"

    elif not thumbIsOpen and indexIsOpen and middelIsOpen and ringIsOpen and pinkyIsOpen:
        return "FOUR!"

    elif not thumbIsOpen and indexIsOpen and middelIsOpen and ringIsOpen and not pinkyIsOpen:
        return "THREE!"

    elif not thumbIsOpen and indexIsOpen and middelIsOpen and not ringIsOpen and not pinkyIsOpen:
        return "TWO!"

    elif not thumbIsOpen and indexIsOpen and not middelIsOpen and not ringIsOpen and not pinkyIsOpen:
        return "ONE!"

    elif not thumbIsOpen and indexIsOpen and not middelIsOpen and not ringIsOpen and pinkyIsOpen:
        return "ROCK!"

    elif thumbIsOpen and indexIsOpen and not middelIsOpen and not ringIsOpen and pinkyIsOpen:
        return "SPIDERMAN!"

    elif not thumbIsOpen and not indexIsOpen and not middelIsOpen and not ringIsOpen and not pinkyIsOpen:
        return "FIST!"






# For webcam input:
cap = cv2.VideoCapture(args.webcam_id)

with mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1) as hands:

  while cap.isOpened() and not rospy.is_shutdown():
    success, image = cap.read()

    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        
        mp_drawing.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        #retrieve handedness of the image ('Left' or 'Right')
        handedness = MessageToDict(results.multi_handedness[0])['classification'][0]['label']

        #get the relevant hand points (we use the wrist as the origin, and normalize to length one)
        handpoints = np.zeros((4,3))
        wrist = landmark_to_array(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST])
        for i, point in enumerate(points_of_interest):
            p = landmark_to_array(hand_landmarks.landmark[point]) - wrist
            p = p / np.linalg.norm(p, ord=2)
            handpoints[i, :] = p

        roll, pitch, yaw = get_angles_from_rotation(np.linalg.pinv(unrotated) @ handpoints)

        gestureID = simpleGesture(hand_landmarks.landmark, handedness)

        gesture_publisher.publish(gestureID)

    cv2.imshow('MediaPipe Hands', image)
    
    key = cv2.waitKey(5)
    if  key & 0xFF == 27: #escape key
        break
    elif key & 0xFF == ord('c'):
        unrotated = handpoints
        unrotated_hand = handedness
        np.save('calib.npy', unrotated)
        print('CALIBRATED!')
    
    r.sleep()

cap.release()


