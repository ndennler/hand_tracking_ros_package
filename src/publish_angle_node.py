#/usr/bin/python3

#----------------------------------------------------
#
#       parses input arguments
#       The only requirement is the webcam id
#
#----------------------------------------------------

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("webcam_id", type=int,
                    help="The number of the webcam device to open. It should be an integer (0 is the laptop's built-in webcam).")

args = parser.parse_args()


import cv2
import numpy as np
from numpy.lib.function_base import angle
import mediapipe as mp
import rospy
from std_msgs.msg import Float64
import json 
from google.protobuf.json_format import MessageToDict
from util import landmark_to_array, get_angles_from_rotation


'''
Initialize the ROS node and set up some 
'''
rospy.init_node('thumb_node')



mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

angle_publisher = rospy.Publisher('thumb_angle', Float64, queue_size=10)

#these are the points used in estimating the rotation matrix of the hand
points_of_interest = [
            mp_hands.HandLandmark.INDEX_FINGER_MCP,
            mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
            mp_hands.HandLandmark.RING_FINGER_MCP,
            mp_hands.HandLandmark.PINKY_MCP,
        ]

unrotated_hand = 'None' #stores the handedness
unrotated = np.array([[-0.60397809, -0.10909924, -0.78949846],
 [-0.39785473, -0.16591713, -0.90232096],
 [-0.16918771, -0.15877018, -0.97271144],
 [ 0.03837452, -0.09477445, -0.99475887]]) #stores the "0" position

last_reading = np.nan #previous reading of thumb angle
decay = 0.5 #parameter for exponential smoohing filter (weights total of all past observatoins with new observation)
r = rospy.Rate(50) # 50hz


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

        #combine reading with history of readings to reduce noise
        last_reading = last_reading * decay + yaw * (1-decay)

        #if some overflow occurs, just re-zero it
        if np.isnan(last_reading):
            last_reading = 0

        # if we haven't calibrated or the hand has changed, we will need to calibrate
        if handedness != unrotated_hand:
            print('Please run the calibration by pressing "c"')

        #otherwise we can publish the angle!
        else:  
            if handedness == 'Left':
                #left hand just publishes angle
                angle_publisher.publish(last_reading)
            else:
                #right hand publishes negative angle since it is mirrored
                angle_publisher.publish(-last_reading)

    cv2.imshow('MediaPipe Hands', image)
    
    key = cv2.waitKey(5)
    if  key & 0xFF == 27: #escape key
        break
    elif key & 0xFF == ord('c'):
        unrotated = handpoints
        unrotated_hand = handedness
    
    r.sleep()

cap.release()