import numpy as np

def landmark_to_array(landmark):
    '''
    Converts a LandMark Object from MediaPipe to a numpy array, which is 
    more easily usable later
    '''
    return np.array([landmark.x, landmark.y, landmark.z]) 

def get_angles_from_rotation(matrix):
    '''
    Given a Rotation matrix of the form:

    [
        [R11, R12, R13],
        [R21, R22, R23],
        [R31, R32, R33]
                         ]

    returns the roll pitch and yaw of the transform
    (we are only interested in the yaw)
    '''
    R11, R12, R13 = matrix[0]
    R21, R22, R23 = matrix[1]
    R31, R32, R33 = matrix[2]


    if R31 != 1 and R31 != -1: 
        pitch_1 = -1*np.arcsin(R31)
        pitch_2 = np.pi - pitch_1 
        roll_1 = np.arctan2( R32 / np.cos(pitch_1) , R33 /np.cos(pitch_1) ) 
        roll_2 = np.arctan2( R32 / np.cos(pitch_2) , R33 /np.cos(pitch_2) ) 
        yaw_1 = np.arctan2( R21 / np.cos(pitch_1) , R11 / np.cos(pitch_1) )
        yaw_2 = np.arctan2( R21 / np.cos(pitch_2) , R11 / np.cos(pitch_2) ) 

        # IMPORTANT NOTE here, there is more than one solution but we choose the first for this case for simplicity !
        # You can insert your own domain logic here on how to handle both solutions appropriately (see the reference publication link for more info). 
        pitch = pitch_1 
        roll = roll_1
        yaw = yaw_1 
    else: 
        yaw = 0 # anything (we default this to zero)
        if R31 == -1: 
            pitch = np.pi/2 
            roll = yaw + np.arctan2(R12,R13) 
        else: 
            pitch = -np.pi/2 
            roll = -1*yaw + np.arctan2(-1*R12,-1*R13) 

    # convert from radians to degrees
    roll = roll*180/np.pi 
    pitch = pitch*180/np.pi
    yaw = yaw*180/np.pi 

    rxyz_deg = [roll , pitch , yaw]
    return rxyz_deg 
