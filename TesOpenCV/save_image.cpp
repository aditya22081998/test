#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main()
{
    VideoCapture cap(2);
    cap.set(CAP_PROP_FRAME_WIDTH,640);
    cap.set(CAP_PROP_FRAME_HEIGHT,480);
    cap.set(CAP_PROP_FPS,10);

    Mat save_img;

    cap >> save_img;

    char Esc = 0;

    while (Esc != 27 && cap.isOpened()) {
        bool Frame = cap.read(save_img);
        if (!Frame || save_img.empty()) {
            cout << "error: frame not read from webcam\n";
            break;
        }
        resizeWindow("imgOriginal", 320, 240);
        namedWindow("imgOriginal", WINDOW_NORMAL);
        imshow("imgOriginal", save_img);
        Esc = waitKey(1);
    }
    imwrite("test9.jpg",save_img);
}