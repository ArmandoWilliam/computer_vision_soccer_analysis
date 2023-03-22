# -*- coding: utf-8 -*-
import time
import cv2 as cv
import numpy as np
from custom_tracker import Tracker
from goal_line_finder import *
import os
# import multiprocessing
# import numba as nb

import kivyApp

#please provide the paths for resources.
yolomodel = {"config_path":"configuration_files/yolo-obj.cfg",
              "model_weights_path":"configuration_files/yolo-obj_best.weights",
              "dataset_names":"configuration_files/obj.names",
              "confidence_threshold": 0.5,
              "threshold":0.3
             }
             
#video_src = "highlights.mp4"
#video_src = "highlights.mp4"

net = cv.dnn.readNetFromDarknet(yolomodel["config_path"], yolomodel["model_weights_path"])
labels = open(yolomodel["dataset_names"]).read().strip().split("\n")

#cv better performances
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

np.random.seed(12345)
layer_names = net.getLayerNames()
layer_names = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

bbox_colors = np.random.randint(0, 255, size=(len(labels), 3))
maxLost = 5
tracker = Tracker(maxLost = maxLost)
#cap = cv.VideoCapture(video_src)

def count_nonblack_np(img): 
    return img.any(axis=-1).sum()

def color_detection(image, show = False): #<-- True for debugging

    # boundaries = [([17, 15, 100], [50, 56, 200]), #orange
    # ([0, 0, 0], [255, 255, 60])] #blacks

    #boundaries = [([0, 0, 200], [0, 0, 150]), #red
    #([240, 240, 240], [250, 250, 250])] #white
    
    #-------------------------- BGR to HSV converter ---------------------------#
    # blue = np.uint8([[[255,0,0]]])
    # hsv_blue = cv.cvtColor(blue,cv.COLOR_BGR2HSV)
    # hsv_blue = hsv_blue.reshape(3)
    # hsv_blue_upper = [hsv_blue[0]+10, 255, 100]
    # hsv_blue_lower = [hsv_blue[0]-10, 100, 100]

    # white = np.uint8([[[255,255,255]]])
    # hsv_white = cv.cvtColor(white,cv.COLOR_BGR2HSV)
    # hsv_white = hsv_white.reshape(3)
    # hsv_white_upper = [hsv_white[0]+10, 255, 255]
    # hsv_white_lower = [hsv_white[0]-10, 100, 100]
    #---------------------------------------------------------------------------#
    
    # Convert image from BGR to HSV
    #hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    boundaries = [([60,30,25], [90,70,60]), #blue team
        ([146,116,96], [200,205,190])] #white team
    
    i = 0
    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")

        try:
            # Threshold the image to get only blue colors
            mask = cv.inRange(image, lower, upper)
            # Bitwise-AND mask and original image
            output = cv.bitwise_and(image, image, mask = mask)
            tot_pix = count_nonblack_np(image)
            color_pix = count_nonblack_np(output)
        except:
            # print("strange things..")
            return 'not_sure'
        ratio = color_pix/tot_pix
        # print("ratio is:", ratio)
        if ratio > 0.01 and i == 0:
            # print("Chelsea")
            return 'ratio is orange'
        elif ratio > 0.01 and i == 1:
            # print("Manchester City")
            return 'black'

        i += 1

        if show:
            cv.imshow("images", np.hstack([image, output]))
            if cv.waitKey(0) & 0xFF == ord('q'):
              cv.destroyAllWindows()
              
    return 'not_sure'

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

score_team_left = 0
score_team_right = 0

def save_image(img):
    global score_team_left
    global score_team_right

    filename = f'GOAL_{score_team_left}_{score_team_right}.jpg'
    cv.imwrite(filename, img)
    print("file name: ", filename)  
        
def goal_team(img, goal_coordinates):

    # Define reference line (e.g., a vertical line passing through the center of the image)
    ref_line = (img.shape[1] // 2, 0), (img.shape[1] // 2, img.shape[0])

    # Draw reference line on image (for visualization purposes)
    cv.line(img, ref_line[0], ref_line[1], (0, 255, 0), 2)
    
    point = tuple(goal_coordinates[0])

    # Check if point is on the left or right side of the reference line
    if point[0] < ref_line[0][0]:
        print("the goal is on the left part of the screen")
        return 0 # point is on the left
    else:
        print("the goal is on the right part of the screen")
        return 1 # point is on the right
        
        
in_goal_area = False
start_time = 0
goal = 0
frames_in_goal = 0


def check_if_is_goal(image, y1, y4, x1, x2, goal_coordinates, ball_center, frames_in_goal_threshold = 4):
    global in_goal_area
    global start_time
    global goal
    global frames_in_goal
    global score_team_left
    global score_team_right
    # save_image(image)
    
    goal_image = image[y1:y4,x1:x2]
    
    if ball_center is not None:
        # ball is detected
        print("ball center: ", ball_center)
        #check if the ball is inside the goal boundig box
        inside_goal_area = cv.pointPolygonTest(goal_coordinates.reshape((-1,1,2)), ball_center, False)
        print(inside_goal_area)
        # check if the ball is in the goal
        if inside_goal_area > 0:
            print("Ball inside goal area")
            goal_line = find_goal_line(goal_image)
            if goal_line is not None:
                print("GOAL LINE DETECTED")
                ball_position = ball_position_relative_to_line(goal_line, ball_center)
                if ball_position is not None:
                    if goal_team(image, goal_coordinates) == 0:
                        # team on the right scored
                        if(ball_position == 'left'):
                            if not in_goal_area:
                                # ball has just entered the goal area
                                in_goal_area = True
                            frames_in_goal += 1
                            if frames_in_goal == 5:
                                score_team_right += 1
                                save_image(image)
                            goal = 1
                    elif goal_team(image, goal_coordinates) == 1:
                        # team on the left scored
                        if(ball_position == 'right'):
                            if not in_goal_area:
                                # ball has just entered the goal area
                                in_goal_area = True
                            frames_in_goal += 1
                            if frames_in_goal == 5:
                                score_team_left += 1
                                save_image(image)
                            goal = 1
        else:
            # ball is detected and is not in the goal area
            print("ball is detected and is not in the goal area")
            in_goal_area = False
            frames_in_goal = 0
            goal = 0
        print("score: ", score_team_left, " - ", score_team_right)
        print("frames in goal: ", frames_in_goal)
        
    return goal

def print_goal_box(image):
    if image is not None:
        # print("GOOOOOOOL")
        # define the position and size of the black box
        box_x = image.shape[1] - 150
        box_y = image.shape[0] - 75
        box_w = 150
        box_h = 75

        # create the black box
        cv.rectangle(image, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)

        # wite "GOAL" in white text on the black box
        text_x = box_x + 10
        text_y = box_y + 50
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 255, 255)
        thickness = 2
        cv.putText(image, "GOAL", (text_x, text_y), font, font_scale, color, thickness)

def print_stopped_game_box(image):
    if image is not None:
        # print("GAME STOPPED")
        # define the position and size of the black box
        box_x = image.shape[1] - 300
        box_y = 0
        box_w = 300
        box_h = 60

        # create the black box for game stopped
        cv.rectangle(image, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)

        # wite "GAME STOPPED" in white text on the black box
        text_x = box_x + 10
        text_y = box_y + 40
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 255, 255)
        thickness = 2
        cv.putText(image, "GAME STOPPED", (text_x, text_y), font, font_scale, color, thickness)

prev_frame = None
total_possession_time_team_A = 0
start_time_team_A = None
ball_center = None
goal_coordinates = None

def process_one_image(image, fps):
    
    # cv.imshow("IMAGE TO PROCESS", image)
    # if cv.waitKey(0) & 0xFF == ord('q'):
    #     cv.destroyAllWindows()

    global prev_frame
    global total_possession_time_team_A
    global start_time_team_A
    (H, W) = image.shape[:2]

    blob = cv.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections_layer = net.forward(layer_names)
    detections_bbox = []
    boxes, confidences, classIDs, player_speed_list, ball_speed_list = [], [], [], [], []
    ball_speed = None

    # Loop over the detected objects
    for out in detections_layer:
        for detection in out:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > yolomodel['confidence_threshold']:
                # convert the coordinates of the bounding box from a realtive scale to a pixel scale taking in consideration the H,W of the image
                box = detection[0:4] * np.array([W, H, W, H])
                # convert from float to int, center coordinates and width and height in pixel of each box
                (centerX, centerY, width, height) = box.astype("int")
                # find the top-left corner coordinates of the box (with the int casting)
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                # add the predicted class label og the detected object to the classIDs
                classIDs.append(classID)
                
    # perform a Non-Maximum Suppression on the bounding boxes and corresponding confidence scores returned by the YOLO model
    # output: list of indices corresponding to the remaining bounding boxes after NMS has been applied
    idxs = cv.dnn.NMSBoxes(boxes, confidences, yolomodel["confidence_threshold"], yolomodel["threshold"])

    j = 0
    k = 0
    result = None
    global ball_center
    global goal_coordinates
    if len(idxs)>0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            x1, y1 = boxes[i][0], boxes[i][1]
            x2, y2 = boxes[i][0] + boxes[i][2], boxes[i][1]
            x3, y3 = boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]
            x4, y4 = boxes[i][0], boxes[i][1] + boxes[i][3]
            detections_bbox.append((x, y, x+w, y+h))

            # assigning color to the bounding boxes, out is a list of the three integers representing the RGB values
            clr = [int(c) for c in bbox_colors[classIDs[i]]]
 
            if labels[classIDs[i]] == "B":
                # the object is the ball
                # Get the center point of the soccer ball bounding box
                ball_center = ((2 * x + w) / 2, (2 * y + h) / 2)
                # print_stopped_game_box(image)
                # print("found the ball")
                if prev_frame is not None:
                    # ball_speed_list.append(float(calculate_object_speed(prev_frame, image, x, y, w, h, 300, fps)))
                    # k+=1
                    #ball_speed = (float(optical_flow_calculate_object_speed(prev_frame, image, x, y, w, h, 400, fps)))
                    cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    # if ball_speed is not None:
                    # cv.putText(image, "{}: {:.4f}: {:.2f} pixel/frame".format(labels[classIDs[i]], confidences[i], ball_speed), (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (300, 0, 0), 2)
                    # if ball_speed == 0:
                    #     # print("game stopped")
                    #     print_stopped_game_box(image)
                else:
                    cv.putText(image, "{}: {:.4f}".format(labels[classIDs[i]], confidences[i]), (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (300, 0, 0), 2)

            if labels[classIDs[i]] == "Goal":
                # the object is the Goal
                # calculate the goal polygon
                # print("found the goal")
                goal_coordinates = np.array([(x1,y1), (x2,y2), (x3,y3), (x4,y4)])
                # check if ball and goal coordinates are detected
                goal = check_if_is_goal(image, y1, y4, x1, x2,  goal_coordinates, ball_center)
                #goal = 1 #debug
                if goal == 1:
                    print_goal_box(image)
                cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # if the object is a player
            # action_and_false_goal.mp4
            if labels[classIDs[i]] == "P":
                # if prev_frame is not None:
                    # print('checking the speed of the player %d' % j)
                    # player_speed_list.append(float(calculate_object_speed(prev_frame, image, boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], 300, 25)))
                    # j+=1
                # print('bounding box num. %d' % i)
                color = color_detection(image[y:y+h,x:x+w])
                if color != 'not_sure':
                    if color == 'black':
                        cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 0), 2)
                    else:
                        cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    #calculate possession time
                    """ if ball_center is not None:
                        player_coordinates = np.array([(x1,y1), (x2,y2), (x3,y3), (x4,y4)])
                        if check_if_is_goal(player_coordinates, ball_center) >= 0:
                            if start_time_team_A is None:
                                start_time_team_A = time.time
                            if start_time_team_A is not None:
                                total_possession_time_team_A = time.time() - start_time_team_A
                                total_possession_time_team_A += total_possession_time_team_A
                                start_time_team_A = None
                                print(total_possession_time_team_A) """
                # if len(player_speed_list) >= 1:
                   # cv.putText(image, "{}: {:.4f}: {:.2f} km/h".format(labels[classIDs[i]], confidences[i], player_speed_list[j-1]), (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (300, 0, 0), 2)
            else:
               cv.rectangle(image, (x, y), (x+w, y+h), clr, 2)
            cv.putText(image, "{}: {:.4f}".format(labels[classIDs[i]], confidences[i]), (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)        


    objects = tracker.update(detections_bbox)

    for (objectID, centroid) in objects.items():
        text = "ID {}".format(objectID)
        cv.putText(image, text, (centroid[0] - 10, centroid[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv.circle(image, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    prev_frame = image.copy()

    return image, result


if __name__ == '__main__':
    kivyApp.SoccerApp().run()