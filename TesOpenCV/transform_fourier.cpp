#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core.hpp>


using namespace cv;
using namespace std;

void calculateDFT(Mat &scr, Mat &dst)
{
    Mat planes[] = { scr, Mat::zeros(scr.size(), CV_32F) };
    Mat complexImg;

    merge(planes, 2, complexImg);

    dft(complexImg, complexImg);
    dst = complexImg;
}

void fftshift(const Mat &input_img, Mat &output_img)
{
    output_img = input_img.clone();
    int cx = output_img.cols / 2;
    int cy = output_img.rows / 2;
    Mat q1(output_img, Rect(0, 0, cx, cy));
    Mat q2(output_img, Rect(cx, 0, cx, cy));
    Mat q3(output_img, Rect(0, cy, cx, cy));
    Mat q4(output_img, Rect(cx, cy, cx, cy));

    Mat temp;
    q1.copyTo(temp);
    q4.copyTo(q1);
    temp.copyTo(q4);
    q2.copyTo(temp);
    q3.copyTo(q2);
    temp.copyTo(q3);

}


int main(){
    Mat image = imread("2benda.jpeg");
    Mat gray;
    Size s(400,300);
    resize(image,image,s);

    imshow("Citra Asli", image);
    image.convertTo(image, CV_32F);

    cvtColor(image, gray, COLOR_BGR2GRAY);

    Mat padded;
    int optimal_row= getOptimalDFTSize 	(gray.rows);
    int optimal_col= getOptimalDFTSize 	(gray.cols);
    copyMakeBorder(gray, padded, 0, optimal_row-gray.rows, 0, optimal_col-gray.cols, BORDER_CONSTANT, Scalar::all(0) );

    Mat DFT_image;
    calculateDFT(padded, DFT_image);

    Mat real, imaginary;
    Mat planes[] = { real, imaginary };

    split(DFT_image, planes);
    Mat mag_image;
    magnitude(planes[0], planes[1], mag_image);

    // switch to a logarithmic scale
    mag_image += Scalar::all(1);
    log(mag_image, mag_image);
    mag_image = mag_image(Rect(0, 0, mag_image.cols & -2, mag_image.rows & -2));

    Mat shifted_DFT;
    fftshift(mag_image, shifted_DFT);

    normalize(shifted_DFT, shifted_DFT, 0, 1, NORM_MINMAX);

    imshow("DFT", shifted_DFT);
    waitKey(0);
    return 0;
}