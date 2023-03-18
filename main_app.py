# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
from custom_tracker import Tracker
# import multiprocessing
# import numba as nb

import kivyApp

#please provide the paths for resources.
yolomodel = {"config_path":"configuration_files/yolo-obj.cfg",
              "model_weights_path":"configuration_files/yolo-obj_best.weights",
              "dataset_names":"configuration_files/obj.names",
              "confidence_threshold": 0.1,
              "threshold":0.05
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

    boundaries = [([17, 15, 100], [50, 56, 200]), #orange
    ([0, 0, 0], [255, 255, 60])] #black

    #boundaries = [([0, 0, 200], [0, 0, 150]), #red
    #([240, 240, 240], [250, 250, 250])] #white
    
    i = 0
    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")

        try:
            mask = cv.inRange(image, lower, upper)
            output = cv.bitwise_and(image, image, mask = mask)
            tot_pix = count_nonblack_np(image)
            color_pix = count_nonblack_np(output)
        except:
            # print("strange things..")
            return 'not_sure'
        ratio = color_pix/tot_pix
        # print("ratio is:", ratio)
        if ratio > 0.01 and i == 0:
            return 'ratio is orange'
        elif ratio > 0.01 and i == 1:
            return 'black'

        i += 1

        if show:
            cv.imshow("images", np.hstack([image, output]))
            if cv.waitKey(0) & 0xFF == ord('q'):
              cv.destroyAllWindows()
    return 'not_sure'

# only if the camera is fixed!
def calculate_player_speed(prev_frame, actual_frame, x, y, w, h, displacement_threshold, fps):
    # initialize the Kalman filter
    kalman = cv.KalmanFilter(4, 2, 0)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.array([[1e-4, 0, 0, 0], [0, 1e-4, 0, 0], [0, 0, 2.5e-2, 0], [0, 0, 0, 2.5e-2]], np.float32) 

    prev_gray = cv.cvtColor(prev_frame, cv.COLOR_RGB2GRAY)
    actual_gray = cv.cvtColor(actual_frame, cv.COLOR_RGB2GRAY)

    # convert the images to 8-bit depth
    prev_gray = cv.convertScaleAbs(prev_gray)
    actual_gray = cv.convertScaleAbs(actual_gray)

    # calculate the optical flow
    prev_corners = np.array([[x, y], [x+w, y], [x, y+h], [x+w, y+h]], dtype=np.float32).reshape(-1, 1, 2)
    next_corners, status, _ = cv.calcOpticalFlowPyrLK(prev_gray, actual_gray, prev_corners, None)

    # update the Kalman filter with the measured position
    if status.all():
        measured = np.array([[next_corners.mean(axis=0)[0][0]], [next_corners.mean(axis=0)[0][1]]], np.float32)
        kalman.correct(measured)

    # predict the position and velocity using the Kalman filter
    prediction = kalman.predict()

    # calculate the speed of the player
    displacement = np.sqrt((prediction[0][0]-x)**2 + (prediction[1][0]-y)**2)
    if displacement < displacement_threshold:
        return 0
    else:
        speed_pixel_frame = displacement
        speed_km_h = (speed_pixel_frame * 0.010 * fps * 3.6) / 100
        return speed_km_h

prev_frame = None

def process_one_image(image):
    global prev_frame
    (H, W) = image.shape[:2]

    blob = cv.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections_layer = net.forward(layer_names)
    detections_bbox = []
    boxes, confidences, classIDs, player_speed = [], [], [], []

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
    result = None
    if len(idxs)>0:
        ball_center = None
        goal_coordinates = None
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            detections_bbox.append((x, y, x+w, y+h))

            # assigning color to the bounding boxes, out is a list of the three integers representing the RGB values
            clr = [int(c) for c in bbox_colors[classIDs[i]]]
 
            if labels[classIDs[i]] == "B":
                # the object is the ball
                ball_center = ((2 * boxes[i][0] + boxes[i][2]) / 2, (2 * boxes[i][1] + boxes[i][3]) / 2)
                # print("found the ball")
                cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

            if labels[classIDs[i]] == "Goal":
                # the object is the Goal
                # calculate the goal polygon
                # print("found the goal")
                x1, y1 = boxes[i][0], boxes[i][1]
                x2, y2 = boxes[i][0] + boxes[i][2], boxes[i][1]
                x3, y3 = boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]
                x4, y4 = boxes[i][0], boxes[i][1] + boxes[i][3]
                goal_coordinates = np.array([(x1,y1), (x2,y2), (x3,y3), (x4,y4)])
                cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # check if ball and goal coordinates are detected
            if ball_center is not None and goal_coordinates is not None:
                print("I'll check if is goal")
                points = goal_coordinates.reshape((-1,1,2))
                result = cv.pointPolygonTest(points, ball_center, False)
                #result = 1 #debug
                if result > 0:
                    print("GOOOOOOOL")
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


            # if the object is a player
            if labels[classIDs[i]] == "P":
                if prev_frame is not None:
                    print('checking the speed of the player %d' % j)
                    player_speed.append(float(calculate_player_speed(prev_frame, image, boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], 500, 25)))
                    j+=1
                print('bounding box num. %d' % i)
                color = color_detection(image[y:y+h,x:x+w])
                if color != 'not_sure':
                    if color == 'black':
                        cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 0), 2)
                    else:
                        cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                if len(player_speed) >= 1:
                    cv.putText(image, "{}: {:.4f}: {:.2f} km/h".format(labels[classIDs[i]], confidences[i], player_speed[j-1]), (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (250, 0, 0), 2)
            else:
               cv.rectangle(image, (x, y), (x+w, y+h), clr, 2)
            cv.putText(image, "{}: {:.4f}".format(labels[classIDs[i]], confidences[i]), (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)        


    objects = tracker.update(detections_bbox)

    for (objectID, centroid) in objects.items():
        text = "ID {}".format(objectID)
        cv.putText(image, text, (centroid[0] - 10, centroid[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv.circle(image, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    prev_frame = image.copy().astype(np.float32)

    return image, result


if __name__ == '__main__':
    kivyApp.SoccerApp().run()