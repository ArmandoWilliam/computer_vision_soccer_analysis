import numpy as np
import cv2 as cv

# when the camera is moving
def optical_flow_calculate_object_speed(prev_frame, next_frame, x, y, w, h, displacement_threshold, fp, show = True):
    prev_frame_float_32 = np.array(prev_frame, np.float32)
    # initialize the Kalman filter
    kalman = cv.KalmanFilter(4, 2, 0)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.array([[1e-4, 0, 0, 0], [0, 1e-4, 0, 0], [0, 0, 2.5e-2, 0], [0, 0, 0, 2.5e-2]], np.float32) 

    prev_gray = cv.cvtColor(prev_frame_float_32, cv.COLOR_BGR2GRAY)
    next_gray = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)

    # convert the images to 8-bit depth
    prev_gray = cv.convertScaleAbs(prev_gray)
    next_gray = cv.convertScaleAbs(next_gray)

    # calculate the optical flow
    prev_corners = np.array([[x, y], [x+w, y], [x, y+h], [x+w, y+h]], dtype=np.float32).reshape(-1, 1, 2)
    next_corners, status, err = cv.calcOpticalFlowPyrLK(prev_gray, next_gray, prev_corners, None)

    # update the Kalman filter with the measured position
    if status.all():
        measured = np.array([[next_corners.mean(axis=0)[0][0]], [next_corners.mean(axis=0)[0][1]]], np.float32)
        kalman.correct(measured)

    # predict the position and velocity using the Kalman filter
    prediction = kalman.predict()

    # calculate the speed of the ball
    corners = np.array([[x, y], [x+w, y], [x, y+h], [x+w, y+h]], dtype=np.float32)
    speed_sum = 0
    for corner in corners:
        displacement = np.sqrt((prediction[0][0]-corner[0])**2 + (prediction[1][0]-corner[1])**2)
        if displacement < displacement_threshold:
            speed = 0
        else:
            speed = displacement
        speed_sum += speed
    speed_pixel_frame = speed_sum / 4
    # speed_km_h = (speed_pixel_frame * 3.6) / 1000
    return speed_pixel_frame