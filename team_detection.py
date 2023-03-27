import numpy as np
import cv2 as cv

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