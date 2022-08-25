#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int main() {
    
Mat image,gray,blurImage,edgeCanny, drawing, zeros, dilate;
cv::Size kernel(5,5);
int blockSize = 2;
int apertureSize = 3;
double k = 0.04;

VideoCapture cap(0);

if (!cap.isOpened()) {

cout << "cannot open camera";

}

while (true) {

cap >> image;
cvtColor(image, gray, CV_BGR2GRAY);
GaussianBlur (gray, blurImage, kernel, 0);
Canny(blurImage,edgeCanny, 40,200);
cv::dilate(edgeCanny, dilate, Mat(), Point(-1,-1), 1, 1, 0);

Mat image_copy= dilate.clone();

Mat dst= Mat::zeros (image_copy.size(), CV_32FC1); 
cornerHarris(image_copy, dst, blockSize, apertureSize, k);

Mat dst_norm, dst_norm_scaled;


cv::normalize( dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, Mat() );
cv::convertScaleAbs( dst_norm, dst_norm_scaled );


for (int i=0; i<dst_norm.rows; i++){
	for( int j = 0; j < dst_norm.cols; j++ )
        {
            if( (int) dst_norm.at<float>(i,j) > 200 )
            {
                cv::circle( dst_norm_scaled, cv::Point(j,i), 5,  cv::Scalar(255), 2, 8, 0 );
            }
        }
}

imshow ("Corner harris", dst_norm_scaled);
imshow("Edge Detection", image_copy);
waitKey(25);

}

return 0;

}
