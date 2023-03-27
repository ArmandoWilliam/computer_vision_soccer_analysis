import cv2 as cv
import math
import numpy as np

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
        # print("the goal is on the left part of the screen")
        return 0 # point is on the left
    else:
        # print("the goal is on the right part of the screen")
        return 1 # point is on the right

in_goal_area = False
goal = 0
frames_in_goal = 0

def check_if_is_goal(image, y1, y4, x1, x2, goal_coordinates, ball_center, frames_in_goal_threshold = 4):
    global in_goal_area
    global goal
    global frames_in_goal
    global score_team_left
    global score_team_right
    # save_image(image)
    
    goal_image = image[y1:y4,x1:x2]
    
    if ball_center is not None:
        # ball is detected
        # print("ball center: ", ball_center)
        #check if the ball is inside the goal boundig box
        inside_goal_area = cv.pointPolygonTest(goal_coordinates.reshape((-1,1,2)), ball_center, False)
        # print(inside_goal_area)
        # check if the ball is in the goal
        if inside_goal_area > 0:
            # print("Ball inside goal area")
            goal_line = find_goal_line(goal_image)
            if goal_line is not None:
                # print("GOAL LINE DETECTED")
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
            # print("ball is detected and is not in the goal area")
            in_goal_area = False
            frames_in_goal = 0
            goal = 0
        # print("score: ", score_team_left, " - ", score_team_right)
        # print("frames in goal: ", frames_in_goal)
        
    return goal

def find_goal_line(img):
    
    # boundaries = [150,255,200], [220,255,255] # barcellona-real
    mccity_boundaries = [100, 190, 190], [190,255,244] # mccity goal
    chelsea_boundaries = [100, 150, 170], [160, 210, 190] # chelsea goal
    
    (lower, upper) = mccity_boundaries
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")
        
    try:
        # Threshold the image to get only blue colors
        mask = cv.inRange(img, lower, upper)
        # Bitwise-AND mask and original image
        output = cv.bitwise_and(img, img, mask = mask)
        # cv.imshow('bitwise mask', output)
        # if cv.waitKey(0) & 0xFF == ord('q'):
        #     cv.destroyAllWindows()
    except:
        # print("strange things..")
        return 'not_sure'
    
    # Convert the image to grayscale
    gray = cv.cvtColor(output, cv.COLOR_BGR2GRAY)
    # cv.imshow('gray image', gray)
    # if cv.waitKey(0) & 0xFF == ord('q'):
    #     cv.destroyAllWindows()
        
    # Apply Gaussian blur to reduce noise
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    # cv.imshow('gaussian noise reduction', blur)
    # if cv.waitKey(0) & 0xFF == ord('q'):
    #     cv.destroyAllWindows()

    # Apply edge detection using the Canny algorithm
    edges = cv.Canny(blur, 150, 250)

    # Use the Hough Line Transform to detect straight lines in the image
    # Define the Hough Line Transform parameters
    rho = 1
    theta = math.pi/180
    threshold = 20
    lines = cv.HoughLinesP(edges, rho,theta, threshold, minLineLength=30, maxLineGap=30)

    # Find the line with the longest length that is approximately vertical
    max_length = 0
    max_line = None
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = ((x2-x1)**2 + (y2-y1)**2)**0.5
            angle = abs(y2-y1)/abs(x2-x1+1e-8)
            if angle < 0.5 and length > max_length:
                max_length = length
                max_line = line

        # Draw the line on the original image
        if max_line is not None:
            x1, y1, x2, y2 = max_line[0]
            cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Display the result
        # cv.imshow('Result', img)
        # cv.imshow('Canny', edges)
        # if cv.waitKey(0) & 0xFF == ord('q'):
        #     cv.destroyAllWindows()
    else:
        print("No lines were detected in the image.")
        
    return max_line

def distance_goal_line_ball(img, goal_line, ball_center):
    distance = None
    ball_x, ball_y = ball_center
    if goal_line is not None:
        # Get the endpoints of the max_line
        x1, y1, x2, y2 = goal_line[0]
        
        # Calculate the distance between the center of the soccer ball and the line
        distance = abs((y2-y1)*ball_x - (x2-x1)*ball_y + x2*y1 - y2*x1) / ((y2-y1)**2 + (x2-x1)**2)**0.5
        if distance is not None:
            print("Distance between the goal_line and the soccer ball: ", distance)
    return distance
        
def get_line_equation(x1, y1, x2, y2):
    slope = (y2 - y1) / (x2 - x1 + 1e-8)
    intercept = y1 - slope * x1
    return slope, intercept

def get_point_position_relative_to_line(point, x1, x2):
    x, y = point
    # slope, intercept = line_equation
    # position = (y - slope * x - intercept) / ((slope ** 2 + 1) ** 0.5)
    mid_x = (x1 + x2) / 2
    position = point[0] - mid_x
    position
    if position > 0:
        return 'right'
    elif position < 0:
        return 'left'
    else:
        return 'on the line'

def ball_position_relative_to_line(goal_line, ball_center):
    ball_x, ball_y = ball_center
    if goal_line is not None:
        # Get the endpoints of the line
        x1, y1, x2, y2 = goal_line[0]
        
        # Get the line equation
        line_equation = get_line_equation(x1, y1, x2, y2)
        
        # Get the position of the ball relative to the line
        position = get_point_position_relative_to_line((ball_x, ball_y), x1, x2)
        # print("Position of the ball relative to the line: ", position)
        return position
    else:
        print("Goal line not detected")
        return None
        
        
if __name__ == '__main__':
    path = r'goal_chelsea.png'
    
    # Load the image
    img = cv.imread(path)
    
    find_goal_line(img)