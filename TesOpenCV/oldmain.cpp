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


using namespace cv;
using namespace std;


static double get_distance(double width_object){
    double known_width=20;
    double focal_lenght=715.0;
    double distance_object=(known_width*focal_lenght)/width_object;
    return distance_object;
}

void setLabel_Distance(Mat& image, double distance){
    int fontface= FONT_HERSHEY_SIMPLEX;
    double scale=0.4;
    int thickness=1;

    string distance_text= to_string(distance);

    Point bawah_kanan(500,460);
    putText(image,distance_text,bawah_kanan,fontface,scale,CV_RGB(255,255,255),thickness,8);
    putText(image,"Distance: ", Point(400,460),fontface,scale, CV_RGB(255,255,255),thickness,LINE_8);

}

void setLabel_FPS(Mat& image, double fps){
    int fontface= FONT_HERSHEY_SIMPLEX;
    double scale=0.4;
    int thickness=1;

    string fps_var=to_string(fps);

    Point atas(40,460);
    putText(image,fps_var,atas,fontface,scale,CV_RGB(255,255,255),thickness,8);
    putText(image,"FPS: ", Point(1,460),fontface,scale, CV_RGB(255,255,255),thickness,LINE_8);
}

static double angle(Point pt1, Point pt2, Point pt0){
    double dx1=pt1.x-pt0.x;
    double dx2=pt2.x-pt0.x;
    double dy1=pt1.y-pt0.y;
    double dy2=pt2.y-pt0.y;

    double distance= (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2));

    return(distance);
}

void setLabel(cv::Mat& im, const std::string label, std::vector<cv::Point>& contour)
{
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

    circle(im, centerPoint, 5, Scalar(255), 2, 8, 0);
    cv::rectangle(im, pt + cv::Point(0, baseline), pt + cv::Point(text.width, -text.height), CV_RGB(255,255,255), FILLED);
    cv::putText(im, center, pt, fontface, scale, CV_RGB(0,0,0), thickness, 8);
}


int main(){
    Mat frame;
    VideoCapture capture("outcpp.avi");

    int fpsCamera=30;
    int fpsCapture=10;

    double fps2=capture.get(CAP_PROP_FPS);
    cout<<"FPS2:"<<fps2<<endl;


    if (capture.isOpened()== false) {
        cout << "ERROR! Unable to open camera\n";
        cin.get();
        return -1;
    }
    chrono::time_point<chrono::high_resolution_clock>
            prev_frame_time(chrono::high_resolution_clock::now());
    chrono::time_point<chrono::high_resolution_clock>
            new_frame_time;

    capture.set(CAP_PROP_FRAME_WIDTH,640);
    capture.set(CAP_PROP_FRAME_HEIGHT,480);

    capture.set(CAP_PROP_FPS,30);

    while (true)
    {
        Mat Image;
        capture>>Image;
        if (Image.empty()) {
            cout << "ERROR! blank frame \n";
            cin.get();
            break;
        }
        new_frame_time = chrono::high_resolution_clock::now();
        chrono::duration<double> duration1(new_frame_time - prev_frame_time);
        double fps = 1/duration1.count();
        setLabel_FPS(Image,fps);

        Mat kernel_identitas= (Mat_<double>(3,3)<< 0,0,0,0,1,0,0,0,0);
        Mat image_identitas;
        filter2D(Image,image_identitas,-1,kernel_identitas,Point(-1,-1),0,4);

        Mat imageContrast;
        image_identitas.convertTo(imageContrast,-1,1.0,-50);

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

        Mat image_copy= edge_canny.clone();
        vector<vector<Point>>contours;

        findContours(image_copy,contours,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);

        vector<Point>approx;

        Mat drawing= Mat::zeros(image_copy.size(),CV_8UC3);

        for(unsigned int i=0;i<contours.size();i++){
            double peri= arcLength(Mat(contours[i]),true);
            approxPolyDP(Mat(contours[i]),approx,peri*0.02, true);

            if(std::fabs(contourArea(contours[i]))<100||!isContourConvex(approx))
                continue;

            if(approx.size()==4){
                int vtc=approx.size();

                vector<double>cos;
                for(int j=2; j<vtc+1; j++){
                    cos.push_back(angle(approx[j%vtc],approx[j-2],approx[j-1]));

                    std::sort(cos.begin(),cos.end());

                    double mincos= cos.front();
                    double maxcos= cos.back();

                    if (vtc==4 && mincos >= -0.1 && maxcos <= 0.3){
                        Rect r= boundingRect(contours[i]);
                        double ratio=std::abs(1-(double)r.width/r.height);
                        cout<<"Width="<< r.width<<endl;
                        cout<<"Focal Length="<< (r.width*40)/20.0<<endl;
                        double distance=get_distance(r.width);
                        setLabel_Distance(drawing, distance);
                        cout<<"Distance="<<distance<<"Centimeter"<<endl;
                        setLabel(drawing, ratio<=0.02 ? "Square": "Rectangle",contours[i]);
                        setLabel_FPS(drawing,fps);
                        drawContours(drawing,contours,(int)i,Scalar(255),2,LINE_8,approx,0);
                    }
                }
            }
        }

        if (duration1.count()>1/fpsCapture){
            prev_frame_time=new_frame_time;
            imshow("Live", imageContrast);
            imshow("Hasil", drawing);
        }

        if (waitKey(1000/fpsCamera)%256 == 27)
            break;
    }
    capture.release();
    return 0;
}
