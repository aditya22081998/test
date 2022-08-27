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

void setLabel(cv::Mat& im, const std::string label, std::vector<cv::Point>& contour)
{
    int fontface = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 0.4;
    int thickness = 1;
    int baseline = 0;

    cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
    cv::Rect r = cv::boundingRect(contour);

    cv::Point pt(r.x + ((r.width - text.width) / 2), r.y + ((r.height + text.height) / 2));
    cv::rectangle(im, pt + cv::Point(0, baseline), pt + cv::Point(text.width, -text.height), CV_RGB(255,255,255), CV_FILLED);
    cv::putText(im, label, pt, fontface, scale, CV_RGB(0,0,0), thickness, 8);
}

//~ void findCornerHarris(Mat image_clone){
//~ int blockSize = 2;
//~ int apertureSize = 3;	
//~ double k = 0.04;
	
	//~ Mat dst= Mat::zeros (image_clone.size(), CV_32FC1); 
				//~ cornerHarris(image_clone, dst, blockSize, apertureSize, k);

				//~ Mat dst_norm, dst_norm_scaled;

				//~ cv::normalize( dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, Mat() );
				//~ cv::convertScaleAbs( dst_norm, dst_norm_scaled );

				//~ for (int i=0; i<dst_norm.rows; i++){
					//~ for( int j = 0; j < dst_norm.cols; j++ ){
						//~ if( (int) dst_norm.at<float>(i,j) > 100 ){
							//~ cv::circle( dst_norm_scaled, cv::Point(j,i), 5,  cv::Scalar(255), 2, 8, 0 );
						//~ }
					//~ }
				//~ }
	//~ imshow ("Corner harris", dst_norm_scaled);
//~ }

int main() {
    
Mat image,gray,blurImage,edgeCanny, drawing, zeros, dilate;
cv::Size kernel(5,5);


VideoCapture cap(0);

double fps = cap.get(CV_CAP_PROP_FPS);
cout << "Frames per second using video.get(CAP_PROP_FPS) : " << fps << endl;

if (!cap.isOpened()){

cout << "cannot open camera";

}

while (true) {

cap >> image;

cvtColor(image, gray, CV_BGR2GRAY);

GaussianBlur(gray, blurImage, kernel, 0);

Canny(blurImage,edgeCanny, 40,200);

cv::dilate(edgeCanny, dilate, Mat(), Point(-1,-1), 1, 1, 0);

Mat image_copy= dilate.clone();
std::vector<std::vector<cv::Point> > contours;

findContours(image_copy, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

std::vector<cv::Point> approx;

Mat pointContour= image.clone();

//~ Mat drawing= Mat zeros (image_copy.size(), CV_8UC3);

for (unsigned int i= 0; i<contours.size(); i++){
		// Approximate contour with accuracy proportional
		// to the contour perimeter
		cv::approxPolyDP(cv::Mat(contours[i]), approx, cv::arcLength(cv::Mat(contours[i]), true)*0.02, true);
		
		// Skip small or non-convex objects 
		if (std::fabs(cv::contourArea(contours[i])) < 100 || !cv::isContourConvex(approx))
			continue;
		
		if (approx.size() == 4){
			// Number of vertices of polygonal curve
			int vtc = approx.size();
			// Get the degree (in cosines) of all corners
			std::vector<double> cos;
			for (int j = 2; j < vtc+1; j++)
				cos.push_back(angle(approx[j%vtc], approx[j-2], approx[j-1]));
			// Sort ascending the corner degree values
			std::sort(cos.begin(), cos.end());

			// Get the lowest and the highest cosine
			double mincos = cos.front();
			double maxcos = cos.back();

			// Use the degrees obtained above and the number of vertices
			// to determine the shape of the contour
			if (vtc == 4 && mincos >= -0.1 && maxcos <= 0.3){
				// Detect rectangle or square
			    cv::Rect r = cv::boundingRect(contours[i]);
				double ratio = std::abs(1 - (double)r.width / r.height);
				setLabel(pointContour, ratio <= 0.02 ? "SQUARE" : "RECTANGLE", contours[i]);
			 }
		}	    
	}
//findCornerHarris(image_copy);
imshow("Edge Detection", image_copy);
imshow("Rectangle", pointContour);
if (waitKey(25)== (0x20))
	break;
}

return 0;

}
