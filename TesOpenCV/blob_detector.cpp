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
#include <opencv2/features2d.hpp>


using namespace cv;
using namespace std;

int main()
{
    Mat im = imread("square2.jpg");

    // Read image

    Size s(400,300);
    resize(im,im,s);
    imshow("original",im);

    if (!im.empty()){
        vector<KeyPoint>kp;

        SimpleBlobDetector::Params param;

//        param.filterByCircularity = true;
//        param.minCircularity = 0.8;
//
//        param.filterByInertia = false;
//        param.minInertiaRatio = 0.01;
//
//
//        param.filterByColor= true;
//        param.minThreshold=10;
//        param.maxThreshold=255;


        param.filterByArea= true;
        param.minArea=150;


        Ptr<SimpleBlobDetector>detector=SimpleBlobDetector::create(param);

        detector->detect(im,kp);
        Mat output;

        drawKeypoints(im,kp,output,Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        imshow("Hasil",output);

    }


//// Setup SimpleBlobDetector parameters.
//    SimpleBlobDetector::Params params;
//
//// Change thresholds
//    params.minThreshold = 10;
//    params.maxThreshold = 256;
//
//// Filter by Area.
//    params.filterByArea = true;
//    params.minArea =50;
//    params.maxArea=700;
//
//// filter my min distance
////params.minDistBetweenBlobs=100;
//
//// Filter by Circularity
//    params.filterByCircularity = true;
//    params.minCircularity = 0.8;
//
//// Filter by Convexity
//    params.filterByConvexity = true;
//    params.minConvexity = 0.5;
//
//// Filter by Inertia
//    params.filterByInertia = false;
//    params.minInertiaRatio = 0.01;
//
////filter by colour
//    params.filterByColor=true;
//    params.blobColor=255;
//
//// Storage for blobs
//    vector<KeyPoint> keypoints;
//
//
//
//// Set up detector with params
//    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
//
//// Detect blobs
//    detector->detect( im, keypoints);
//
////the total no of blobs detected are:
//    for(size_t i=0;i<keypoints.size();i++)
//        cout<<keypoints[i].size<<endl;
//
//
////location of first blob
//    Point2f point1=keypoints.at(0).pt;
//    float x1=point1.x;
//    float y1=point1.y;
//    cout<<"location of the first blob is "<<x1<<","<<y1;
//
//
//// Draw detected blobs as red circles.
//// DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures
//// the size of the circle corresponds to the size of blob
//
//    Mat im_with_keypoints;
//    drawKeypoints( im, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
//
//// Show blobs
//    imshow("keypoints", im_with_keypoints );
//
    waitKey();
    destroyAllWindows();
}