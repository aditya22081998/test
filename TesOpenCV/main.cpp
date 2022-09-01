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


using namespace cv;
using namespace std;

static double angle(cv::Point pt1, cv::Point pt2, cv::Point pt0)
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

static double distance(cv::Point pt1, cv::Point pt2, cv::Point pt0)
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return sqrt(pow(dx2 - dx1, 2) + pow(dy2 - dy1, 2) * 1.0);

}

void setLabel(cv::Mat& im, const std::string label, std::vector<cv::Point>& contour)
{
    int fontface = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 0.4;
    int thickness = 1;
    int baseline = 0;

    Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
    Rect r = cv::boundingRect(contour);
    Point pt(r.x + ((r.width - text.width) / 2), r.y + ((r.height + text.height) / 2));
    Point centerPoint(r.x + r.width/2, r.y+r.height/2);
//    for(int i=0; i<centerPoint.rows();i++){
//        for(int j=0; j<centerPoint.cols();j++){
//        double distance= matchShapes(centerPoint(0,0),centerPoint(0,1), CV_CONTOURS_MATCH_I1,0)
//        cout<<distance;
//        }
//
//    }


    // cv::Point A(r.x,r.y);
	// cv::Point B(r.x+r.width,r.y);
	// cv::Point C(r.x+r.width,r.y+r.height);
	// cv::Point D(r.x,r.y+r.height);

	// circle(im, A, 5, Scalar(255), 2, 8, 0);
	// circle(im, B, 5, Scalar(255), 2, 8, 0);
	// circle(im, C, 5, Scalar(255), 2, 8, 0);
	// circle(im, D, 5, Scalar(255), 2, 8, 0);

     circle(im, centerPoint, 5, Scalar(255), 2, 8, 0);

//    cv::rectangle(im, pt + cv::Point(0, baseline), pt + cv::Point(text.width, -text.height), CV_RGB(255,255,255), CV_FILLED);
//    cv::putText(im, label, pt, fontface, scale, CV_RGB(0,0,0), thickness, 8);
}

int main() {

Mat image, gray, blurImage, edgeCanny, drawing, zeros, dilate,dilate2, erode, M_Closing;
Size kernel(5,5);

VideoCapture cap(0);

cap.set (CV_CAP_PROP_FPS, 10);

double fps = cap.get(CV_CAP_PROP_FPS);

cout << "Frames per second using video.get(CAP_PROP_FPS) : " << fps << endl;

cap.set(CAP_PROP_FRAME_WIDTH, 320);
cap.set(CAP_PROP_FRAME_HEIGHT, 240);

if (!cap.isOpened()){

cout << "cannot open camera";

}

while (true) {

cap >> image;

cvtColor(image, gray, CV_BGR2GRAY);

//cv::dilate(gray, dilate, Mat(), Point(-1,-1), 1, 1, 0) ;

GaussianBlur(gray, blurImage, kernel, 0);

//cv::erode(blurImage, erode, Mat(), Point(-1,-1), 1, 1, 0 );

morphologyEx(blurImage, M_Closing, CV_MOP_CLOSE, Mat(), Point(-1,-1), 1, 1, 0  );

Canny(M_Closing, edgeCanny, 40,200);

Mat image_copy= edgeCanny.clone();

std::vector<std::vector<Point> > contours;

findContours(image_copy, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

std::vector<Point> approx;

//~ Mat pointContour= image.clone();

Mat drawing= Mat::zeros (image_copy.size(), CV_8UC3);

//Mat drawingcenter = Mat::zeros (drawing.size(), CV_8UC3);
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
                cout<<distance(approx[j%vtc], approx[j-2], approx[j-1])<<endl;
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
//				cv::Point centerPoint(r.x + r.width/2, r.y+r.height/2);
//				int PC=centerPoint.size();
//
//				for(size_t k=2; k<PC+1; k++){
////                    cv::approxPolyDP(cv::Mat(contours[i]), PC, peri*0.02, true);
//                    int center=centerPoint.size();
//                    cout<<distance(PC[k%center], PC[k-2], PC[k])<<endl;
//				}

				setLabel(drawing, ratio <= 0.02 ? "SQUARE" : "RECTANGLE", contours[i]);
				drawContours( drawing, contours, (int)i, Scalar(255), 2, LINE_8, approx, 0 );
			 }
		}
	}
imshow("Drawing Rectangle",drawing);
imshow("Edge Detection", image_copy);
// imshow("Drawing Center", drawingcenter);
if (waitKey(25)== (0x20))
	break;
}

return 0;

}
