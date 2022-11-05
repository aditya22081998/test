#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <time.h>
#include <bits/stdc++.h>
#include <string>
#include <boost/lexical_cast.hpp>


using namespace cv;
using namespace std;

static double angle(cv::Point pt1, cv::Point pt2, cv::Point pt0)
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2));
}

static double focal_length_find(double measure_distance, double real_width,double frame_width){
    double  focal_length = (frame_width*measure_distance)/real_width;
    return focal_length;
}

static double distance_find(double Focal_Length, double real_width_object,double frame_width_object){
    double  distance = (real_width_object*Focal_Length)/frame_width_object;
    return distance;
}

static double euclideanDist(cv::Point a, cv::Point b)
{
    cv::Point diff = a - b;
    return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}



void setLabel(cv::Mat& im, const std::string label, std::vector<cv::Point>& contour)
{
    int fontface = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 0.4;
    int thickness = 1;
    int baseline = 0;

//    Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
    Rect r = cv::boundingRect(contour);
    Point centerPoint(r.x + r.width / 2, r.y + r.height / 2);
    double distance = euclideanDist(centerPoint);
    cout << distance;

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
        cv::rectangle(im, bawah + cv::Point(0, baseline), bawah + cv::Point(teks.width, -teks.height), CV_RGB(255,255,255), FILLED);
        cv::putText(im, kiri, bawah, fontface, scale, CV_RGB(0,0,0), thickness, 8);
    } else if((r.x + r.width / 2)>213 && (r.x + r.width / 2)<=426){
        Size teks = cv::getTextSize(tengah, fontface, scale, thickness, &baseline);
        Point bawah(10, 20 );
        cv::rectangle(im, bawah + cv::Point(0, baseline), bawah + cv::Point(teks.width, -teks.height), CV_RGB(255,255,255), FILLED);
        cv::putText(im, tengah, bawah, fontface, scale, CV_RGB(0,0,0), thickness, 8);
    } else if((r.x + r.width / 2)>426 && (r.x + r.width / 2)<=640){
        Size teks = cv::getTextSize(kanan, fontface, scale, thickness, &baseline);
        Point bawah(10, 20 );
        cv::rectangle(im, bawah + cv::Point(0, baseline), bawah + cv::Point(teks.width, -teks.height), CV_RGB(255,255,255), FILLED);
        cv::putText(im, kanan, bawah, fontface, scale, CV_RGB(0,0,0), thickness, 8);
    }

//    string center= to_string(centerPoint);
    Size text = cv::getTextSize(center, fontface, scale, thickness, &baseline);
    Point pt(r.x + ((r.width - text.width) / 2), r.y + ((r.height + text.height) / 2));

    circle(im, centerPoint, 5, Scalar(255), 2, 8, 0);
    cv::rectangle(im, pt + cv::Point(0, baseline), pt + cv::Point(text.width, -text.height), CV_RGB(255,255,255), FILLED);
    cv::putText(im, center, pt, fontface, scale, CV_RGB(0,0,0), thickness, 8);
}

int main() {

    Mat image, gray, blurImage, edgeCanny, drawing, zeros, M_Closing;
    Size kernel(5,5);

    double known_distance=50;

    double known_width= 25;

    VideoCapture cap("video1.mp4");

    cap.set (CAP_PROP_FPS, 10);

    double fps = cap.get(CAP_PROP_FPS);

    cout << "Frames per second using video.get(PROP_FPS) : " << fps << endl;

    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);

    if (!cap.isOpened()){

        cout << "cannot open camera";

    }

    while (true) {

        cap >> image;
        Mat kernel_identitas= (Mat_<double> (3,3) << 0,0,0,0,1,0,0,0,0);
        Mat image_identitas;
        filter2D(image,image_identitas,-1,kernel_identitas,Point(-1,-1),0,4);

        Mat sharp_image;
        Mat kernel_sharp= (Mat_<double>(3,3)<< 0,-1,0,-1,5,-1,0,-1,0);

        filter2D(image_identitas, sharp_image, -1, kernel_sharp, Point(-1,-1),0, BORDER_DEFAULT);

        cvtColor(sharp_image, gray, COLOR_BGR2GRAY);

        GaussianBlur(gray, blurImage, kernel, 0);

        morphologyEx(blurImage, M_Closing, MORPH_CLOSE, Mat(), Point(-1,-1), 1, 1, 0  );

        Canny(M_Closing, edgeCanny, 40,200);

        Mat image_copy= edgeCanny.clone();

        std::vector<std::vector<Point> > contours;

        findContours(image_copy, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        std::vector<Point> approx;

        Mat drawing= Mat::zeros (image_copy.size(), CV_8UC3);

        for (unsigned int i= 0; i<contours.size(); i++){
            // Approximate contour with accuracy proportional
            // to the contour perimeter
            double peri = arcLength(Mat(contours[i]), true);

            cv::approxPolyDP(Mat(contours[i]), approx, peri*0.02, true);

            // Skip small or non-convex objects
            if (std::fabs(contourArea(contours[i])) < 100 || !isContourConvex(approx))
                continue;

            if (approx.size() == 4){
                // Number of vertices of polygonal curve
                int vtc = approx.size();
                // Get the degree (in cosines) of all corners
                std::vector<double> cos;
                for (int j = 2; j < vtc+1; j++){
                    cos.push_back(angle(approx[j%vtc], approx[j-2], approx[j-1]));
//                cout<<distance(approx[j%vtc], approx[j-2], approx[j-1])<<endl;
                }

                // Sort ascending the corner degree values
                std::sort(cos.begin(), cos.end());

                // Get the lowest and the highest cosine
                double mincos = cos.front();
                double maxcos = cos.back();

                // Use the degrees obtained above and the number of vertices
                // to determine the shape of the contour
                if (vtc == 4 && mincos >= -0.1 && maxcos <= 0.3){
                    // Detect rectangle or square
                    Rect r = boundingRect(contours[i]);
                    double ratio = std::abs(1 - (double)r.width / r.height);
                    double face_width= r.width;
                    double FocalLength= focal_length_find(known_distance,known_width,face_width);
                    if (face_width!=0){
                        double  Distance_Object= distance_find(FocalLength,known_width,face_width);
                        cout<< Distance_Object<<endl;
                    }

                    setLabel(drawing, ratio <= 0.02 ? "SQUARE" : "RECTANGLE", contours[i]);
                    drawContours( drawing, contours, (int)i, Scalar(255), 2, LINE_8, approx, 0 );
                }
            }
        }
        imshow("Drawing Rectangle",drawing);
        imshow("Edge Detection", image_copy);
        imshow("Sharp Image", sharp_image);
        if (waitKey(25)== (0x20))
            break;
    }

    return 0;

}
