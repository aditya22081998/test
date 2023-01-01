//
// Created by aditya on 23/11/22.
//
#include "opencv2/imgcodecs.hpp"
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <chrono>
#include <ctime>
#include <string>
#include <vector>
#include <boost/lexical_cast.hpp>

using namespace std;
using namespace cv;

Mat gray, edge, Image, image_copy,drawing;

static double angle(Point pt1, Point pt2, Point pt0){
    double dx1=pt1.x-pt0.x;
    double dx2=pt2.x-pt0.x;
    double dy1=pt1.y-pt0.y;
    double dy2=pt2.y-pt0.y;

    double distance= (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2));

    return(distance);
}

void draw_and_fill_contours(vector<vector<Point>>& contours,vector<vector<Point>>& hull,vector<Vec4i>& hierarchy) {
    Mat contours_result = drawing.clone();
    Mat fill_contours_result = Mat::zeros(drawing.size(), CV_8UC3);

    for (unsigned int i = 0; i < contours.size(); ++i) {
        Scalar color = Scalar(0, 0, 255);
        drawContours(contours_result, contours, i, color, 4, 8, hierarchy, 0, cv::Point());
    }

    fillPoly(fill_contours_result, hull, cv::Scalar(255, 255, 255), LINE_8, 0);

//    int morph_size = 2;
//    Mat element = getStructuringElement(
//    MORPH_RECT, Size(2 * morph_size + 1,
//    2 * morph_size + 1),
//    Point(morph_size, morph_size));
//    Mat dilasi;
//    dilate(fill_contours_result,dilasi,element,Point(-1,-1),1);

    Mat close;
    morphologyEx(fill_contours_result,close,MORPH_CLOSE, Mat(),Point(-1,-1),1,1,0);

    resizeWindow("Contours Result", 320, 240);
    resizeWindow("Fill Contours Result", 320, 240);
    namedWindow("Contours Result", WINDOW_NORMAL);
    namedWindow("Fill Contours Result", WINDOW_NORMAL);
    imshow("Contours Result",contours_result);
    imshow("Fill Contours Result",close);
}

//static double euclideanDist(Point mc, Point contours){
//
//    for(unsigned  int i=0; i<contours.size(); i++){
//        Point2f diff = mc[i+1] - mc[i];
//        return sqrt(diff.x*diff.x + diff.y*diff.y);
//    }
//}
void set_Label(Mat& im, vector<Point>& contour, std::string label ){
    int fontface = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 0.4;
    int thickness = 1;
    int baseline = 0;

    Rect r = cv::boundingRect(contour);
    Point centerPoint(r.x + r.width / 2, r.y + r.height / 2);
//    double distance = euclideanDist(centerPoint);
//    cout << distance;

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
    Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
    Point pt(r.x + ((r.width - text.width) / 2), r.y + ((r.height + text.height) / 2)-25);

//    circle(im, centerPoint, 5, Scalar(255), 2, 8, 0);
    cv::rectangle(im, pt + cv::Point(0, baseline), pt + cv::Point(text.width, -text.height), CV_RGB(255,255,255), FILLED);
    cv::putText(im, label, pt, fontface, scale, CV_RGB(0,0,0), thickness, 8);

}

void set_pointcenter(Mat& im, vector<Point>& contour){
    int fontface = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 0.4;
    int thickness = 1;
    int baseline = 0;

//    Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
    Rect r = cv::boundingRect(contour);
    Point centerPoint(r.x + r.width / 2, r.y + r.height / 2);
//    double distance = euclideanDist(centerPoint);
//    cout << distance;

    string center= boost::lexical_cast<string>(centerPoint);


//    string center= to_string(centerPoint);
    Size text = cv::getTextSize(center, fontface, scale, thickness, &baseline);
    Point pt(r.x + ((r.width - text.width) / 2), r.y + ((r.height + text.height) / 2));

    circle(im, centerPoint, 5, Scalar(255), 2, 8, 0);
    cv::rectangle(im, pt + cv::Point(0, baseline), pt + cv::Point(text.width, -text.height), CV_RGB(255,255,255), FILLED);
    cv::putText(im, center, pt, fontface, scale, CV_RGB(0,0,0), thickness, 8);

}

int main (int argc, char** argv){

    VideoCapture capture(2);
//    Image = imread(argv[1], IMREAD_COLOR);

    if (capture.isOpened()== false) {
        cout << "ERROR! Unable to open camera\n";
        cin.get();
        return -1;
    }
    capture.set(CAP_PROP_FRAME_WIDTH,640);
    capture.set(CAP_PROP_FRAME_HEIGHT,480);
    capture.set(CAP_PROP_FPS,10);

    while (true) {
        capture>>Image;
        if (Image.empty()) {
            cout << "ERROR! blank frame \n";
            cin.get();
            break;
        }
        Mat sharpening_image;
        Mat kernel_sharpening= (Mat_<double>(3,3)<<  -1 ,-1, -1 ,-1, 9, -1, -1, -1, -1);
        filter2D(Image,sharpening_image,-1,kernel_sharpening,Point(-1,-1),0, 4);

        cvtColor(sharpening_image,gray, COLOR_BGR2GRAY);

        Mat gaussian_blur;
        Size kernel(5,5);
        GaussianBlur(gray , gaussian_blur, kernel, 1,1);

        Mat morph_close;
        morphologyEx(gaussian_blur,morph_close,MORPH_CLOSE, Mat(),Point(-1,-1),1,1,0);

        Canny(morph_close,edge,100,200);

        image_copy= edge.clone();
        vector<vector<Point>>contours;
        vector<Vec4i> hierarchy;
        findContours(image_copy,contours,hierarchy,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);

        vector<Point>approx;

        drawing= Mat::zeros(image_copy.size(),CV_8UC3);
        vector<vector<Point>> hull(contours.size());
        vector<Moments>mu(contours.size());

        for(unsigned int i=0;i<contours.size();i++){
            mu[i]= cv::moments(contours[i], false);
        }

        vector<Point2f>mc(contours.size());
        for (unsigned int i = 0; i <contours.size() ; i++) {
            mc[i]=Point2f (mu[i].m10/mu[i].m00,mu[i].m01/mu[i].m00);

        }
//        euclideanDist(mc, contours);

        for(unsigned int i=0; i<contours.size();i++){

            convexHull(Mat(contours[i]),hull[i],false);
            double peri= arcLength(Mat(contours[i]),true);
            approxPolyDP(Mat(contours[i]),approx,peri*0.02, true);

            if(std::fabs(contourArea(contours[i]))<100||!isContourConvex(approx))
                continue;

            if(approx.size()==4){
                int vtc=approx.size();

                vector<double>cos;
                for(int j=2; j<vtc+1; j++){
                    cos.push_back(angle(approx[j%vtc],approx[j-2],approx[j-1]));

                    sort(cos.begin(),cos.end());

                    double mincos= cos.front();
                    double maxcos= cos.back();

                    if (vtc==4 && mincos >= -0.1 && maxcos <= 0.3){
                        Rect r= boundingRect(contours[i]);
                        double ratio=std::abs(1-(double)r.width/r.height);
                        drawContours(drawing,contours,(int)i,Scalar(255),2,LINE_8,approx,0);
//                        circle(drawing, mc[i], 8, Scalar(225, 0, 0), -1, 8, 0);
                        set_Label(drawing, contours[i],ratio<=0.02?"Square": "Rectangle");
                        set_pointcenter(drawing, contours[i]);
                    }
                }
            }
        }

        draw_and_fill_contours(contours,hull,hierarchy);

        if (waitKey(25) == 27)
            break;
    }
//    Image.release();
    capture.release();
    return 0;
}

