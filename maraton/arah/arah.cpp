#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <libgen.h>
#include <signal.h>
#include <math.h>
#include <vector>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace cv;
using namespace std;

int main(){

	VideoCapture cap(-1);
	 cap.set(CV_CAP_PROP_FRAME_WIDTH,320);
     cap.set(CV_CAP_PROP_FRAME_HEIGHT,240);
      Mat frame;
     //var arah
     CascadeClassifier lurus,kanan,kiri;
     lurus.load("lurus.xml");
     kanan.load("kanan.xml");
     kiri.load("kiri.xml");

     vector <Rect> right;
     vector <Rect> left;
     vector <Rect> straight;

     char arah='x';
  while(1){
	cap >>frame;
//	cvtColor (frame,frame,COLOR_BGR2GRAY);
 	 kanan.detectMultiScale(frame, right, 1.1, 200, 0);
 	 kiri.detectMultiScale(frame, left, 1.1, 200, 0);
 	 lurus.detectMultiScale(frame, straight, 1.1, 2, 0);
     
     for(size_t i=0;i<right.size();i++){
			 arah='r';
			 Point point(right[i].x,right[i].y);
			 Point point2(right[i].x+right[i].width,right[i].y+right[i].height); 
			 rectangle(frame,point,point2,Scalar(0,255,0),2);
			 }
	
     for(size_t i=0;i<left.size();i++){
			 arah='l';
			 Point point(left[i].x,left[i].y);
			 Point point2(left[i].x+left[i].width,left[i].y+left[i].height); 
			 rectangle(frame,point,point2,Scalar(255,0,0),2);
			 }
    
     for(size_t i=0;i<straight.size();i++){
			 arah='s';
			 Point point(straight[i].x,straight[i].y);
			 Point point2(straight[i].x+straight[i].width,straight[i].y+straight[i].height); 
			 rectangle(frame,point,point2,Scalar(255,0,255),2);
			 }
	imshow("frame", frame);
	
	if(waitKey(30) >= 0) {cout<<arah;break;}

}


}
