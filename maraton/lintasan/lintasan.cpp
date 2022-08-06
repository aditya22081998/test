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
//var citra
Mat hsv;
Mat frame;
Mat mask;
Mat roi;
Point p;
int passing;
Scalar low,high;
Rect c;

//var contour
vector <vector<Point> > contour;
vector <Vec4i> h;
vector <Mat> hp;

//var stat ob	
bool compareContourAreas (std::vector<cv::Point> contour1, std::vector<cv::Point> contour2 )
 {
    double i = fabs( contourArea(cv::Mat(contour1)) );
    double j = fabs( contourArea(cv::Mat(contour2)) );
    return ( i < j );
}

void kali(){

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
       
		if(waitKey(30) >= 0) 
			break;
			
  	}
}

int main(){
	kali();
	VideoCapture cap(-1);
	 cap.set(CV_CAP_PROP_FRAME_WIDTH,320);
     cap.set(CV_CAP_PROP_FRAME_HEIGHT,240);
     passing=0;
  while(1){

		vector <Rect> det;
    	cap >> frame;  
         
         cvtColor (frame ,hsv,COLOR_BGR2HSV);
         //threshold         
         inRange (hsv,low,high,mask);
         //contour
         findContours(mask,contour,h,CV_RETR_LIST,CV_CHAIN_APPROX_NONE);

         //sort kontur 
	if (contour.size()!=0)
		{
		 passing=0;
		 sort(contour.begin(),contour.end(),compareContourAreas);
         drawContours(frame, contour, contour.size()-1, Scalar(0,255,0), 1, 8, h, 0, Point() );

         //detect bentuk
         Moments m= moments(contour[contour.size()-1],1);
         
         try
         {
         	//perhitungan titik tengah
			 p=Point (m.m10/m.m00, m.m01/m.m00);
			
		 }
		 catch(Exception ex){
				cout<<"got error";
				continue;
		 }
            
		//visual titik tengah		
	    circle(frame, p, 2, Scalar(255,0,0), -1);
      
		}
	else
		{
			//kalibrasi ulang selama 100 frame
			passing++;
			rectangle (frame,c,Scalar(0,255,0),1);
			 if (passing>100){
				c=Rect (p.x,p.y,10,10);
				roi=hsv(c);
				Scalar hsvmean=mean(roi);
                low= Scalar(hsvmean[0]-3,hsvmean[1]-70,hsvmean[2]-70);
				high= Scalar(hsvmean[0]+10,hsvmean[1]+150,hsvmean[2]+150);
			 }
		 }

   imshow("frame", frame);
   	cout<<"x : "<<p.x<<" y : "<<p.y<<endl;
    

   if(waitKey(30) >= 0) break;
			
		}
   
        return 0;
}
