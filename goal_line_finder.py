import cv2 as cv
import math
import numpy as np

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
        print("Position of the ball relative to the line: ", position)
        return position
    else:
        print("Goal line not detected")
        return None
        
        
if __name__ == '__main__':
    path = r'goal_chelsea.png'
    
    # Load the image
    img = cv.imread(path)
    
    find_goal_line(img)