#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <time.h>

using namespace cv;
using namespace std;
RNG rng(12345);

int main()
{

    Mat image, gray, blurImage, edgeCanny, drawing, zeros, dilate, dilate2, erode, M_Closing;
    Size kernel(5, 5);

    Mat cap = imread("square2.jpg", IMREAD_COLOR);

    while (true)
    {

        cvtColor(cap, gray, CV_BGR2GRAY);

        // cv::dilate(gray, dilate, Mat(), Point(-1,-1), 1, 1, 0) ;

        GaussianBlur(gray, blurImage, kernel, 0);

        // cv::erode(blurImage, erode, Mat(), Point(-1,-1), 1, 1, 0 );

        morphologyEx(blurImage, M_Closing, CV_MOP_CLOSE, Mat(), Point(-1, -1), 1, 1, 0);

        Canny(M_Closing, edgeCanny, 40, 200);

        Mat image_copy = edgeCanny.clone();

        std::vector<std::vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(image_copy, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

        drawing = Mat::zeros(image_copy.size(), CV_8UC3);
        for (size_t i = 0; i < contours.size(); i++)
        {

            drawContours(drawing, contours, (int)i, Scalar(255), 2, LINE_8, hierarchy, 0);
        }

        imwrite("hasil.jpg", drawing);
    }
}
