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
#include <bits/stdc++.h>
#include <boost/lexical_cast.hpp>

using namespace cv;
using namespace std;

static double angle(cv::Point pt1, cv::Point pt2, cv::Point pt0)
{
    // cout << "X0=" << pt0.x << endl;
    // cout << "X1=" << pt1.x << endl;
    // cout << "X2=" << pt2.x << endl;
    // cout << "Y0=" << pt0.y << endl;
    // cout << "Y1=" << pt1.y << endl;
    // cout << "Y2=" << pt2.y << endl;
    // cout << "width=" << pt0.width;
    double dx1 = pt1.x - pt0.x;
    // cout <<"dx1=" << dx1 << endl;
    double dy1 = pt1.y - pt0.y;
    // cout << "dy1=" << dy1 << endl;
    double dx2 = pt2.x - pt0.x;
    // cout << "dx2=" << dx2 << endl;
    double dy2 = pt2.y - pt0.y;
    // cout << "dy2=" << dy2 << endl;
    return (dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2));
}

static double distance(cv::Point pt1, cv::Point pt2, cv::Point pt0)
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return sqrt(pow(dx2 - dx1, 2) + pow(dy2-dy1, 2) );
}

void setLabel(cv::Mat &im, const std::string label, std::vector<cv::Point> &contour)
{
    int fontface = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 1;
    int thickness = 1;
    int baseline = 0;

//    Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
    Rect r = cv::boundingRect(contour);
    Point centerPoint(r.x + r.width / 2, r.y + r.height / 2);

    string center= boost::lexical_cast<string>(centerPoint);
//    int centerP=(int)center;
//    Point_ <int> Point_ (centerPoint);
//    int centerP= centerPoint;
    string kiri="Kiri";
    string kanan="kanan";
    string tengah="tengah";
    if((r.x + r.width / 2)>0 && (r.x + r.width / 2)<=213){
        Size teks = cv::getTextSize(kiri, fontface, scale, thickness, &baseline);
        Point bawah(10, 20 );
        cv::rectangle(im, bawah + cv::Point(0, baseline), bawah + cv::Point(teks.width, -teks.height), CV_RGB(255,255,255), CV_FILLED);
        cv::putText(im, kiri, bawah, fontface, scale, CV_RGB(0,0,0), thickness, 8);
    } else if((r.x + r.width / 2)>213 && (r.x + r.width / 2)<=426){
        Size teks = cv::getTextSize(tengah, fontface, scale, thickness, &baseline);
        Point bawah(10, 20 );
        cv::rectangle(im, bawah + cv::Point(0, baseline), bawah + cv::Point(teks.width, -teks.height), CV_RGB(255,255,255), CV_FILLED);
        cv::putText(im, tengah, bawah, fontface, scale, CV_RGB(0,0,0), thickness, 8);
    } else if((r.x + r.width / 2)>426 && (r.x + r.width / 2)<=640){
        Size teks = cv::getTextSize(kanan, fontface, scale, thickness, &baseline);
        Point bawah(10, 20 );
        cv::rectangle(im, bawah + cv::Point(0, baseline), bawah + cv::Point(teks.width, -teks.height), CV_RGB(255,255,255), CV_FILLED);
        cv::putText(im, kanan, bawah, fontface, scale, CV_RGB(0,0,0), thickness, 8);
    }

    //    string center= to_string(centerPoint);
    Size text = cv::getTextSize(center, fontface, scale, thickness, &baseline);
    Point pt(r.x + ((r.width - text.width) / 2), r.y + ((r.height + text.height) / 2));
//    Point a(r.x, r.y);
//    circle(im, a ,5, Scalar(255), 2, 8, 0);
//    circle(im, centerPoint, 5, Scalar(255), 2, 8, 0);
    cv::rectangle(im, pt + cv::Point(0, baseline), pt + cv::Point(text.width, -text.height), CV_RGB(255,255,255), CV_FILLED);
    cv::putText(im, center, pt, fontface, scale, CV_RGB(0,0,0), thickness, 8);
}

int main()
{

    Mat image, gray, blurImage, edgeCanny, drawing, zeros, dilate, dilate2, erode, M_Closing;
    Size kernel(5, 5);

    Mat cap = imread("square4.png", IMREAD_COLOR);


    while (true)
    {

        cvtColor(cap, gray, CV_BGR2GRAY );

        GaussianBlur(gray, blurImage, kernel, 0);

        morphologyEx(blurImage, M_Closing, CV_MOP_CLOSE, Mat(), Point(-1, -1), 1, 1, 0);

        Canny(M_Closing, edgeCanny, 100, 200);

        Mat image_copy = edgeCanny.clone();

        std::vector<std::vector<Point>> contours;
        findContours(image_copy, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
        std::vector<Point> approx;

        Mat drawing = Mat::zeros(image_copy.size(), CV_8UC3);
        for (unsigned int i = 0; i < contours.size(); i++)
        {
            // Approximate contour with accuracy proportional
            // to the contour perimeter
            double peri = arcLength(Mat(contours[i]), true);

            cv::approxPolyDP(Mat(contours[i]), approx, peri * 0.02, true);

            // Skip small or non-convex objects
            if (std::fabs(contourArea(contours[i])) < 100 || !isContourConvex(approx))
                continue;

            if (approx.size() == 4)
            {
                // Number of vertices of polygonal curve
                int vtc = approx.size();
                // cout <<vtc;
                // Get the degree (in cosines) of all corners
                std::vector<double> cos;
                for (int j = 2; j < vtc + 1; j++)
                {
                    cos.push_back(angle(approx[j % vtc], approx[j - 2], approx[j - 1]));
                    // cout << approx[j % vtc] << endl
                    //      << approx[j - 2] << endl
                    //      << approx[j - 1] << endl;
                    // cout << "Angle=" << angle(approx[j % vtc], approx[j - 2], approx[j - 1]) << endl;
//                    Point_<int> pixelsPerMetric = 0;
//                    cout<<"Distance Pixel Object="<<distance(approx[j % vtc], approx[j - 2], approx[j - 1]) << endl;
//                    if(pixelsPerMetric=0){
//
//                    }
                }

                // Sort ascending the corner degree values
                std::sort(cos.begin(), cos.end());

                // Get the lowest and the highest cosine
                double mincos = cos.front();
                // cout<<mincos<<endl;
                double maxcos = cos.back();
                // cout<<maxcos<<endl;
                // Use the degrees obtained above and the number of vertices
                // to determine the shape of the contour
                if (vtc == 4 && mincos >= -0.1 && maxcos <= 0.3)
                {
                    // Detect rectangle or square
                    Rect r = boundingRect(contours[i]);
                    // cout<<r;
//                    cout << r.width << endl;
//                    cout << r.height << endl;
                    double ratio = std::abs(1 - (double)r.width / r.height);

                    setLabel(drawing, ratio <= 0.02 ? "SQUARE" : "RECTANGLE", contours[i]);
                    drawContours(drawing, contours, (int)i, Scalar(255), 2, LINE_8, approx, 0);

                }
            }
        }

        //imshow("hasil", image_copy);
        cv::resizeWindow("Display frame", 320, 240);
        cv::resizeWindow("Image", 320, 240);
        namedWindow("Display frame", WINDOW_NORMAL);
        namedWindow("Image", WINDOW_NORMAL);
        imshow("Display frame", drawing);
        imshow("Image", cap);
        if (waitKey(25)== (0x20))
            break;
        void destroyAllWindows();
    }
}

