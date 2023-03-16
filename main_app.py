import cv2 as cv
import numpy as np
from custom_tracker import Tracker

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

def calculate_player_speed(prev_frame, actual_frame, x, y, w, h):
    prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
    act_gray = cv.cvtColor(actual_frame, cv.COLOR_BGR2GRAY)

    # apply a Gaussian blur to the image to reduce noise
    prev_gray = cv.GaussianBlur(prev_gray, (5, 5), 0)
    act_gray = cv.GaussianBlur(act_gray, (5, 5), 0)

    # calculate optical flow between the previous and current frames
    flow = cv.calcOpticalFlowFarneback(prev_gray, act_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # calculate the magnitude and angle of the optical flow vectors
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

    # create a mask to threshold the magnitude of the flow vectors
    mask = np.zeros_like(magnitude, dtype=np.uint8)
    mask[magnitude > 2] = 255

    # find contours in the mask to locate moving objects
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
    # calculate the average magnitude and angle of the flow vectors around the center of the contour
    avg_magnitude = np.mean(magnitude[y:y+h, x:x+w])
    avg_angle = np.mean(angle[y:y+h, x:x+w])
        
    # calculate the speed of the player
    speed = avg_magnitude * np.cos(avg_angle)
    
    # update the previous frame
    # prev_frame = act_gray.copy().astype(np.float32)

    return speed

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
    if len(idxs)>0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            detections_bbox.append((x, y, x+w, y+h))

            # assigning color to the bounding boxes, out is a list of the three integers representing the RGB values
            clr = [int(c) for c in bbox_colors[classIDs[i]]]

            ball_center = None
            goal_coordinates = None 
            if labels[classIDs[i]] == "B":
                # the object is the ball
                ball_center = ((2 * boxes[i][0] + boxes[i][2]) / 2, (2 * boxes[i][1] + boxes[i][3]) / 2)
                # print("found the ball")
            if labels[classIDs[i]] == "Goal":
                # the object is the Goal
                # calculate the goal polygon
                # print("found the goal")
                x1, y1 = boxes[i][0], boxes[i][1]
                x2, y2 = boxes[i][0] + boxes[i][2], boxes[i][1]
                x3, y3 = boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]
                x4, y4 = boxes[i][0], boxes[i][1] + boxes[i][3]
                goal_coordinates = np.array([(x1,y1), (x2,y2), (x3,y3), (x4,y4)])
            
            # check if ball and goal coordinates are detected
            if ball_center is not None and goal_coordinates is not None:
                print("I'll check if is goal")
                points = goal_coordinates.reshape((-1,1,2))
                result = cv.pointPolygonTest(points, ball_center, False)
                if result > 0:
                    print("GOOOOOOOL")

            # if the object is a player
            if labels[classIDs[i]] == "P":
                if prev_frame is not None:
                    print(f'checking the speed of the player {j}')
                    player_speed.append(float(calculate_player_speed(prev_frame, image, boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3])))
                    j+=1
                print(f'bounding box num. {i}')
                color = color_detection(image[y:y+h,x:x+w])
                if color != 'not_sure':
                    if color == 'black':
                        cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 0), 2)
                    else:
                        cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
            else:
                cv.rectangle(image, (x, y), (x+w, y+h), clr, 2)
            if prev_frame is None:
                cv.putText(image, "{}: {:.4f}".format(labels[classIDs[i]], confidences[i]), (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)
            if prev_frame is not None:
                cv.putText(image, "{}: {:.4f}: {:.2f}".format(labels[classIDs[i]], confidences[i], player_speed[j-1]), (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)
            
                

    objects = tracker.update(detections_bbox)

    for (objectID, centroid) in objects.items():
        text = "ID {}".format(objectID)
        cv.putText(image, text, (centroid[0] - 10, centroid[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv.circle(image, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    prev_frame = image.copy().astype(np.float32)
    return image

if __name__ == '__main__':
    kivyApp.SoccerApp().run()