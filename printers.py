import cv2 as cv

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