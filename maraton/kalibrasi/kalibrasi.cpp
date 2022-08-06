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

Mat hsv;
Mat frame;
Mat mask;
Mat roi;
Scalar low,high;
Rect c;

int main(){

	VideoCapture cap(-1);
	 cap.set(CV_CAP_PROP_FRAME_WIDTH,320);
     cap.set(CV_CAP_PROP_FRAME_HEIGHT,240);


	while (1)
 	{	//ambil citra input
		cap >> frame;
		
		//konversi HSV
		cvtColor (frame,hsv,COLOR_BGR2HSV);
		

		//set roi
		c=Rect (frame.size().width/2-10,frame.size().height-60,10,10);
		roi=hsv(c);
		
		//rata-rata nilai roi
		Scalar hsvmean=mean(roi);
		
		//set ambang batas
		low= Scalar(hsvmean[0]-3,hsvmean[1]-10,hsvmean[2]-30);
		high= Scalar(hsvmean[0]+10,hsvmean[1]+100,hsvmean[2]+150);
		
		//out visual
		rectangle (frame,c,Scalar(0,255,0),1);
		imshow("frame",frame);
       
        cout<<"Hue		  : "<<hsvmean[0]<<endl;
        cout<<"Saturation : "<<hsvmean[1]<<endl;
        cout<<"Value	  : "<<hsvmean[2]<<endl;

		if(waitKey(30) >= 0) 
			break;
			
  	}



}
