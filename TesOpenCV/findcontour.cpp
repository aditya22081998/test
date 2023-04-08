#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core.hpp>
#include <cmath>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <time.h>
#include <bits/stdc++.h>
#include <boost/lexical_cast.hpp>
#include <opencv2/plot.hpp>

using namespace cv;
using namespace std;

static double get_distance(double width_object){
    double known_width=20;
    double focal_lenght=715.0;
    double distance_object=(known_width*focal_lenght)/width_object;
    return distance_object;
}

void setLabel_Distance(Mat image, double distance){
    int fontface= FONT_HERSHEY_SIMPLEX;
    double scale=0.4;
    int thickness=1;

    string distance_text= to_string(distance);

    Point bawah_kanan(500,460);
    putText(image,distance_text,bawah_kanan,fontface,scale,CV_RGB(255,255,255),thickness,8);
    putText(image,"Distance: ", Point(400,460),fontface,scale, CV_RGB(255,255,255),thickness,LINE_8);

}


static double angle( cv::Point pt1, cv::Point pt2, cv::Point pt0)
{
     cout << "X0=" << pt0.x << endl;
     cout << "X1=" << pt1.x << endl;
     cout << "X2=" << pt2.x << endl;
     cout << "Y0=" << pt0.y << endl;
     cout << "Y1=" << pt1.y << endl;
     cout << "Y2=" << pt2.y << endl;
    // cout << "width=" << pt0.width;
    double dx1 = pt1.x - pt0.x;
    cout <<"dx1=" << dx1 << endl;
    double dy1 = pt1.y - pt0.y;
    cout << "dy1=" << dy1 << endl;
    double dx2 = pt2.x - pt0.x;
    cout << "dx2=" << dx2 << endl;
    double dy2 = pt2.y - pt0.y;
    cout << "dy2=" << dy2 << endl;
//    cout<<(dx1 * dx2 + dy1 * dy2)<<"/"<<sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2)) <<endl;
    double cos= (dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2))+ 1e-10;
    cout<<"Cosinus="<<cos<<endl;
    return (cos);
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
//    Point a(r.x, r.y);
//    circle(im, a ,5, Scalar(255), 2, 8, 0);
//    circle(im, centerPoint, 5, Scalar(255), 2, 8, 0);
    cv::rectangle(im, pt + cv::Point(0, baseline), pt + cv::Point(text.width, -text.height), CV_RGB(255,255,255), FILLED);
    cv::putText(im, center, pt, fontface, scale, CV_RGB(0,0,0), thickness, 8);
}

int main(int argc, char** argv)
{

    Mat image, gray, blurImage, edgeCanny, drawing, zeros, dilate, dilate2, erode, M_Closing;
    Size kernel(5, 5);

    Mat cap = imread(argv[1], IMREAD_COLOR);


    while (true)
    {

        Mat kernel_identitas= (Mat_<double>(3,3)<< 0,0,0,0,1,0,0,0,0);
        Mat image_identitas;
        filter2D(cap,image_identitas,-1,kernel_identitas,Point(-1,-1),0,4);

        Mat imageContrast;
        image_identitas.convertTo(imageContrast,-1,1,0);

        Mat sharpening_image;
        Mat kernel_sharpening= (Mat_<double>(3,3)<<0,-1,0,-1,5,-1,0,-1,0);
        filter2D(imageContrast,sharpening_image,-1,kernel_sharpening,Point(-1,-1),0, 4);

        Mat gray;
        cvtColor(sharpening_image,gray, COLOR_BGR2GRAY);

        Mat gaussian_blur;
        Size kernel(5,5);
        GaussianBlur(gray , gaussian_blur, kernel, 0);

        Mat morph_close;
        morphologyEx(gaussian_blur,morph_close,MORPH_CLOSE, Mat(),Point(-1,-1),1,1,0);

        Mat edge_canny;
        Canny(morph_close,edge_canny,100,200);

        Mat image_copy = edge_canny.clone();

        std::vector<std::vector<Point>> contours;
        findContours(image_copy, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        std::vector<Point> approx;

        Mat drawing = Mat::zeros(image_copy.size(), CV_8UC3);
        for (unsigned int i = 0; i < contours.size(); i++)
        {
            // Approximate contour with accuracy proportional
            // to the contour perimeter
            double peri = arcLength(Mat(contours[i]), true);

            cv::approxPolyDP(Mat(contours[i]), approx, peri * 0.02, true);

            if (approx.size() == 4 && fabs(contourArea(Mat(approx))) > 1000 && isContourConvex(Mat(approx)))
            {
                // Number of vertices of polygonal curve
                int vtc = approx.size();
                // cout <<vtc;
                // Get the degree (in cosines) of all corners
                std::vector<double> cos;
                for (int j = 2; j < vtc + 1; j++)
                {
                    cos.push_back(angle( approx[j % vtc], approx[j - 2], approx[j - 1]));
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
                cout<<"mincos="<<mincos<<endl;
                double maxcos = cos.back();
                cout<<"maxcos="<<maxcos<<endl;
                // Use the degrees obtained above and the number of vertices
                // to determine the shape of the contour
                if (vtc == 4 && mincos >= -0.1 && maxcos <= 0.3)
                {
                    // Detect rectangle or square
                    Rect r = boundingRect(contours[i]);
                    // cout<<r;
//                    cout << "width="<<r.width << endl;
//                    cout << "height="<<r.height << endl;
                    double ratio = std::abs(1 - (double)r.width / r.height);

                    //cout<<"Width="<< r.width<<endl;
                    //cout<<"Focal Length="<< (r.width*40)/20.0<<endl;
                    double distance= get_distance(r.width);
//                    cout<<"Distance="<<distance<<"Centimeter"<<endl;
                    setLabel_Distance(drawing,distance);
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
        imshow("Display frame", edge_canny);
        imshow("Image", cap);
        if (waitKey(25)== (27))
            break;
        void destroyAllWindows();
    }
}

