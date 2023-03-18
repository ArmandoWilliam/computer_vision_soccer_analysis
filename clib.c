#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <projectEnv/lib/python3.8/site-packages/numpy/core/include/numpy/arrayobject.h>

using namespace cv;

double calculate_speed_for_rect(unsigned char* prev_frame, unsigned char* actual_frame, int x, int y, int w, int h, int width, int height) {
    Mat prev_gray(height, width, CV_8UC1, prev_frame);
    Mat act_gray(height, width, CV_8UC1, actual_frame);

    // apply a Gaussian blur to the image to reduce noise
    GaussianBlur(prev_gray, prev_gray, Size(5, 5), 0);
    GaussianBlur(act_gray, act_gray, Size(5, 5), 0);

    // calculate optical flow between the previous and current frames
    Mat flow;
    calcOpticalFlowFarneback(prev_gray, act_gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

    // calculate the magnitude and angle of the optical flow vectors
    Mat xy[2];
    split(flow, xy);
    Mat magnitude, angle;
    cartToPolar(xy[0], xy[1], magnitude, angle, true);

    // create a mask to threshold the magnitude of the flow vectors
    Mat mask = Mat::zeros(height, width, CV_8UC1);
    threshold(magnitude, mask, 2, 255, THRESH_BINARY);

    // find contours in the mask to locate moving objects
    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;
    findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // calculate the average magnitude and angle of the flow vectors around the center of the contour
    double avg_magnitude = 0.0;
    double avg_angle = 0.0;
    for (int i = y; i < y+h; i++) {
        for (int j = x; j < x+w; j++) {
            if (mask.at<unsigned char>(i,j) != 0) {
                avg_magnitude += magnitude.at<float>(i,j);
                avg_angle += angle.at<float>(i,j);
            }
        }
    }
    avg_magnitude /= (double)countNonZero(mask);
    avg_angle /= (double)countNonZero(mask);

    // calculate the speed of the player
    double speed = avg_magnitude * cos(avg_angle);

    return speed;
}
