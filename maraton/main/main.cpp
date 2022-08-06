#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <libgen.h>
#include <signal.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "Camera.h"
#include "mjpg_streamer.h"
#include "LinuxDARwIn.h"

#ifdef AXDXL_1024
 #define MOTION_FILE_PATH    ((char *)"../../../../Data/motion_1024.bin")
#else
 #define MOTION_FILE_PATH    ((char *)"../../../../Data/motion_4096.bin")
#endif

#define INI_FILE_PATH       ((char *)"../../../../Data/config.ini")
#define M_INI   			((char *)"../../../Data/slow-walk.ini")
#define SCRIPT_FILE_PATH    "script.asc"
#define U2D_DEV_NAME0       "/dev/ttyUSB0"
#define U2D_DEV_NAME1       "/dev/ttyUSB1"

    using namespace cv;
	using namespace std;
    using namespace Robot;
     

int isRunning = 1;
char hec='f';
     char arah='x';
vector<vector<Point> > contours;
CascadeClassifier lurus,kanan,kiri;

void change_current_dir()
    {
        char exepath[1024] = {0};
        if(readlink("/proc/self/exe", exepath, sizeof(exepath)) != -1)
            chdir(dirname(exepath));
    }



bool compareContourAreas (std::vector<cv::Point> contour1, std::vector<cv::Point> contour2 )
 {
    double i = fabs( contourArea(cv::Mat(contour1)) );
    double j = fabs( contourArea(cv::Mat(contour2)) );
    return ( i < j );
}

Point p;
Point2D pos;
Rect c (320/2,240-60,10,10);
Scalar low,high;
BallTracker tracker = BallTracker();
vector <Rect> righty;
vector <Rect> lefty;
vector <Rect> straight;

ofstream testdata;

int main(){
    int passing=0;
	testdata.open ("datamarathon.csv");
	testdata <<"No, waktu proses, perintah \n";
	
	lurus.load("lurus.xml");
	kanan.load("kanan.xml");
	kiri.load("kiri.xml");
	VideoCapture cap(-1);
	cap.set(CV_CAP_PROP_FRAME_WIDTH,320);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT,240);
    //if(!cap.isOpened()) :
    //{
     //   cout<<"fail";
    //    return 0;
    //}

    Mat hsv;
    Mat frame;
    Mat mask;
	Mat roi;
    vector <vector<Point> > contour;
    vector<Vec4i> h;
	vector <Mat> hp;	
		
    printf( "\n===== Walk Backward Tutorial for DARwIn =====\n\n");
     
    change_current_dir();
     
    minIni* ini = new minIni(INI_FILE_PATH);
     
    	//////////////////// Framework Initialize ////////////////////////////
    	LinuxArbotixPro linux_arbotixpro(U2D_DEV_NAME0);
        ArbotixPro arbotixpro(&linux_arbotixpro);
        
    if (MotionManager::GetInstance()->Initialize(&arbotixpro) == false)
        {
            linux_arbotixpro.SetPortName(U2D_DEV_NAME1);
            if (MotionManager::GetInstance()->Initialize(&arbotixpro) == false)
                {
                    printf("Fail to initialize Motion Manager!\n");
                    return 0;
                }
        }
        
			//Create MotionManager object and registers head and walking modules, then timers are initialized.
		
    Walking::GetInstance()->LoadINISettings(ini);
    usleep(100);
    MotionManager::GetInstance()->LoadINISettings(ini);
    MotionManager::GetInstance()->AddModule((MotionModule*)Action::GetInstance());
    MotionManager::GetInstance()->AddModule((MotionModule*)Head::GetInstance());
    MotionManager::GetInstance()->AddModule((MotionModule*)Walking::GetInstance());
    LinuxMotionTimer linuxMotionTimer;
    linuxMotionTimer.Initialize(MotionManager::GetInstance());
    linuxMotionTimer.Start();
    MotionManager::GetInstance()->SetEnable(true);
    
			/////////////////////////////////////////////////////////////////////
     
            /////////////////////////Capture Motor Position//////////////////////
    int n = 0;
    int param[JointData::NUMBER_OF_JOINTS * 5];
    int wGoalPosition, wStartPosition, wDistance;
   
    /*
				Walking::GetInstance()->A_MOVE_AMPLITUDE = 0.0; //default lurus
				Walking::GetInstance()->X_MOVE_AMPLITUDE = 15.0;//default maju
				Walking::GetInstance()->PERIOD_TIME = 1500.0;
				Walking::GetInstance()->Z_MOVE_AMPLITUDE = 30.0;
				Walking::GetInstance()->Y_OFFSET = 50.0;
				Walking::GetInstance()->R_OFFSET = 20.0;
				//Walking::GetInstance()->X_OFFSET = 5.0;
				Walking::GetInstance()->Z_OFFSET = 60.0;
				Walking::GetInstance()->Y_SWAP_AMPLITUDE = 20.0;
				Walking::GetInstance()->Z_SWAP_AMPLITUDE = 3.0;
				Walking::GetInstance()->HIP_PITCH_OFFSET = 8.6;
   
   
				Walking::GetInstance()->Z_MOVE_AMPLITUDE = 30.0;
                Walking::GetInstance()->HIP_PITCH_OFFSET = 8.6;
				Walking::GetInstance()->Y_OFFSET = 75.0;
				Walking::GetInstance()->R_OFFSET = 20.0;
				Walking::GetInstance()->Z_OFFSET = 60.0;
				Walking::GetInstance()->Y_SWAP_AMPLITUDE = 20.0;
				//Walking::GetInstance()->Z_SWAP_AMPLITUDE = 3.0;
				Walking::GetInstance()->PERIOD_TIME = 1500.0;
				Walking::GetInstance()->A_MOVE_AMPLITUDE = 0.0; 
				Walking::GetInstance()->X_MOVE_AMPLITUDE = 15.0;
				//Walking::GetInstance()-> BALANCE_ENABLE = false ;
*/
    for (int id = JointData::ID_R_SHOULDER_PITCH; id < JointData::NUMBER_OF_JOINTS; id++)
        {
            wStartPosition = MotionStatus::m_CurrentJoints.GetValue(id);
            wGoalPosition = Walking::GetInstance()->m_Joint.GetValue(id);
            if ( wStartPosition > wGoalPosition )
                wDistance = wStartPosition - wGoalPosition;
            else
                wDistance = wGoalPosition - wStartPosition;

            wDistance >>= 2;
            
            if ( wDistance < 8 )
                wDistance = 8;

            param[n++] = id;
            param[n++] = ArbotixPro::GetLowByte(wGoalPosition);
            param[n++] = ArbotixPro::GetHighByte(wGoalPosition);
            param[n++] = ArbotixPro::GetLowByte(wDistance);
            param[n++] = ArbotixPro::GetHighByte(wDistance);
        }

     arbotixpro.SyncWrite(AXDXL::P_GOAL_POSITION_L, 5, JointData::NUMBER_OF_JOINTS - 1, param);

           
     
    	printf("Press the ENTER key to begin!\n");
    	
     
        Head::GetInstance()->m_Joint.SetEnableHeadOnly(true, true);
        Walking::GetInstance()->m_Joint.SetEnableBodyWithoutHead(true, true);
    	MotionManager::GetInstance()->SetEnable(true); //Walking and MotionManager enable motors
            
        
        // /*
        //test posisi
        //Walking::GetInstance()->m_Joint.SetValue(3,700.00);
        //Walking::GetInstance()->m_Joint.SetValue(4,400.00);
        
        
        Walking::GetInstance()->m_Joint.SetAngle(3,25);
		Walking::GetInstance()->m_Joint.SetAngle(4,-25);
		 Head::GetInstance()->MoveToHome();
        Head::GetInstance()->MoveByAngle(0,-40);
        
        
        //getchar();
        //return 0;
        // */
        
        //calibrate color

auto startdata=chrono::system_clock::now();
auto enddata=chrono::system_clock::now();
auto start=chrono::system_clock::now();
auto end=chrono::system_clock::now();
std::chrono::duration<double, std::milli> duration = end-start;
std::chrono::duration<double, std::milli> durationdata = enddata-startdata;
int num=1;
int passingsimbol=0;

while (1)
{
    cap >> frame;
    //resize(frame,frame,Size(320,240));
    c=Rect (frame.size().width/2-10,frame.size().height-60,10,10);
    cvtColor (frame,hsv,COLOR_BGR2HSV);
         roi=hsv(c);
         Scalar hsvmean=mean(roi);
		 low= Scalar(hsvmean[0]-3,hsvmean[1]-10,hsvmean[2]-30);
         high= Scalar(hsvmean[0]+10,hsvmean[1]+100,hsvmean[2]+150);
         rectangle (frame,c,Scalar(0,255,0),1);
         imshow("frame",frame);
       
         if(waitKey(30) >= 0) {
			 enddata=chrono::system_clock::now();
			 durationdata = enddata-startdata;
			 testdata<<num<<", "<<chrono::duration_cast<chrono::milliseconds>(enddata-startdata).count()<<", kalibrasi awal\n";
			 
			 break;}
			
  }

int langkah=0;
char det='x';
//time awal
start=chrono::system_clock::now();
double A_amp=0;
for(;;)
{ startdata=chrono::system_clock::now();
	
num++;
  if(Walking::GetInstance()->IsRunning() == false && ((passing<=0)||(det!='x')))
  {
	  Walking::GetInstance()->X_MOVE_AMPLITUDE = 15.0;	
	  Walking::GetInstance()->Start();                 
  }
  
	  
	  
    
    //langkah++;	
    hec='f';
  try{
		
         cap >> frame; 
        // resize(frame,frame,Size(320,240));

     kanan.detectMultiScale(frame, righty, 1.1, 200, 0);
 	 kiri.detectMultiScale(frame, lefty, 1.1, 450, 0);
 	 lurus.detectMultiScale(frame, straight, 1.1, 2, 0);
     if(righty.size()!=0&& det=='x'){
     for(size_t i=0;i<righty.size();i++){
			 arah='r';
			 Point point(righty[i].x,righty[i].y);
			 Point point2(righty[i].x+righty[i].width,righty[i].y+righty[i].height); 
			 rectangle(frame,point,point2,Scalar(0,255,0),2);
			 enddata=chrono::system_clock::now();
			 durationdata = enddata-startdata;
			 testdata<<num<<", "<<chrono::duration_cast<chrono::milliseconds>(enddata-startdata).count()<<", deteksi simbol kanan\n";
			 num++;
			 goto force;
			 }
	}else if (lefty.size()!=0&& det=='x'){
     for(size_t i=0;i<lefty.size();i++){
			 arah='l';
			 Point point(lefty[i].x,lefty[i].y);
			 Point point2(lefty[i].x+lefty[i].width,lefty[i].y+lefty[i].height); 
			 rectangle(frame,point,point2,Scalar(255,0,0),2);
			 enddata=chrono::system_clock::now();
			 durationdata = enddata-startdata;
			 testdata<<num<<", "<<chrono::duration_cast<chrono::milliseconds>(enddata-startdata).count()<<", deteksi simbol kiri\n";num++;
			 goto force;
			 }
    }else if (straight.size()!=0&& det=='x'){
     for(size_t i=0;i<straight.size();i++){
			 arah='s';
			 Point point(straight[i].x,straight[i].y);
			 Point point2(straight[i].x+straight[i].width,straight[i].y+straight[i].height); 
			 rectangle(frame,point,point2,Scalar(255,0,255),2);
			 enddata=chrono::system_clock::now();
			 durationdata = enddata-startdata;
			 testdata<<num<<", "<<chrono::duration_cast<chrono::milliseconds>(enddata-startdata).count()<<" deteksi simbol lurus\n";num++;
			 goto force;
			 }  
		} 
		startdata=chrono::system_clock::now();
		cvtColor (frame ,hsv,COLOR_BGR2HSV);         
        inRange (hsv,low,high,mask);
        enddata=chrono::system_clock::now();
        durationdata = enddata-startdata;
		testdata<<num<<", "<<chrono::duration_cast<chrono::milliseconds>(enddata-startdata).count()<<", segmentasi citra\n";num++;
        startdata=chrono::system_clock::now();
        findContours(mask,contour,h,CV_RETR_LIST,CV_CHAIN_APPROX_NONE);
		
	if (contour.size()!=0&passingsimbol<=0)
		{
			det='x';
		sort(contour.begin(),contour.end(),compareContourAreas);
			 
		if(Walking::GetInstance()->IsRunning() == false)
		{
			Walking::GetInstance()->Start();
           
        }
         passing=0;
         drawContours(frame, contour, contour.size()-1, Scalar(0,255,0), 1, 8, h, 0, Point() );
         Moments m= moments(contour[contour.size()-1],1);
         try{
			 p=Point (m.m10/m.m00, m.m01/m.m00);
			 pos.X=p.x;
			 pos.Y=p.y;
			 }
			catch(Exception ex){
				cout<<"got error";
				continue;
				}
		enddata=chrono::system_clock::now();
		durationdata = enddata-startdata;
			 testdata<<num<<", "<<chrono::duration_cast<chrono::milliseconds>(enddata-startdata).count()<<", deteksi kontur lintasan\n";num++;
		
     			tracker.Process(pos);
         
				
				
         Walking::GetInstance()->X_MOVE_AMPLITUDE = 15.0;
         Walking::GetInstance()->A_MOVE_AMPLITUDE =A_amp;
					 cout<<" cek :"<<Walking::GetInstance()->A_MOVE_AMPLITUDE<<endl;
        //usleep(600000);
         circle(frame, p, 2, Scalar(255,0,0), -1);
      
		}
		else
		{ if(passingsimbol<=0 && det=='x'){continue;}else{
			Walking::GetInstance()->Stop();
			passing++;
			rectangle (frame,c,Scalar(0,255,0),1);
			 if (passing>100){
				 startdata=chrono::system_clock::now();
				c=Rect (p.x,p.y,10,10);
				roi=hsv(c);
				Scalar hsvmean=mean(roi);
                low= Scalar(hsvmean[0]-3,hsvmean[1]-70,hsvmean[2]-70);
				high= Scalar(hsvmean[0]+10,hsvmean[1]+150,hsvmean[2]+150);
				enddata=chrono::system_clock::now();
				durationdata = enddata-startdata;
			 testdata<<num<<", "<<chrono::duration_cast<chrono::milliseconds>(enddata-startdata).count()<<", kalibrasi ulang\n";num++;
		
			 }
		 }}
		 
		 force:
		 if(arah=='s'){
			 Walking::GetInstance()->Stop();
			 passingsimbol++;
			 cout<<"passing : "<<passingsimbol;
			  if (passingsimbol>100){det=arah;

			  Head::GetInstance()->MoveToHome();
			  Head::GetInstance()->MoveByAngle(0,10);
			  passingsimbol=0;
			  usleep(100);
			  }
			 // Head::GetInstance()->MoveByAngle(0,0);
			 //break;
			  
		 }else if(arah=='r'){
			 Walking::GetInstance()->Stop();
			 passingsimbol++;
			 cout<<passingsimbol;
			  if (passingsimbol>100){
				  det=arah;
			 Head::GetInstance()->MoveToHome();
			 Head::GetInstance()->MoveByAngle(-40,10);
			 passingsimbol=0;
			  }
			 //break;
		 }else if(arah=='l'){
			 Walking::GetInstance()->Stop();
			 passingsimbol++;
			 cout<<passingsimbol;
			  if (passingsimbol>100){
				det=arah;

			 Head::GetInstance()->MoveToHome();
			 Head::GetInstance()->MoveByAngle(40,10);
			 passingsimbol=0;
			  }
			 //break;
			 }
			 
			 //set time
			 end=chrono::system_clock::now();
			 //if time < walking period,continue
			cout<<chrono::duration_cast<chrono::milliseconds>(end-start).count()<<endl;
			 if(!(chrono::duration_cast<chrono::milliseconds>(end-start).count() <1500))
			 {//else
				
				 cout<<chrono::duration_cast<chrono::milliseconds>(end-start).count() ;
				 start=chrono::system_clock::now();
				 end=chrono::system_clock::now();
			 if(AXDXL::Value2Angle(MotionStatus::m_CurrentJoints.GetValue(20))<0)
			 {A_amp=(AXDXL::Value2Angle(MotionStatus::m_CurrentJoints.GetValue(20))+35);}
			 else{A_amp=(AXDXL::Value2Angle(MotionStatus::m_CurrentJoints.GetValue(20))-35);}
			 ;
			  cout<<"change to"<<Walking::GetInstance()->A_MOVE_AMPLITUDE<<endl;
		 }
         imshow("frame", frame);
         if(waitKey(30) >= 0) {
			 testdata.close();
			 break;}
			
			
		}

   catch(Exception& e){
			
			cout<< " done ";
			return 0;
			}
		
        if(hec=='t'){
		 }
    	     
     
 }
     
        return 0;
}


