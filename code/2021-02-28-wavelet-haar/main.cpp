#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

// clang++ main.cpp -std=c++11 `pkg-config --libs opencv4` `pkg-config --cflags opencv4` -lstdc++

int main() {
    Mat img = imread("test.jpg", 0);
    int Height = img.cols;
    int Width = img.rows;

    int depth = 2;
    int depthcount = 1;
    Mat tmp = Mat::ones(Width, Height, CV_32FC1);
    Mat wavelet = Mat::ones(Width, Height, CV_32FC1);
    Mat imgtmp = img.clone();
    imgtmp.convertTo(imgtmp, CV_32FC1);
    while (depthcount<=depth){
        Width = img.rows / depthcount;
        Height = img.cols / depthcount;

        for (int i = 0; i < Width; i++){
            for (int j = 0; j < Height / 2; j++){
                tmp.at<float>(i, j) = (imgtmp.at<float>(i, 2 * j) + imgtmp.at<float>(i, 2 * j + 1)) / 2;
                tmp.at<float>(i, j + Height / 2) = (imgtmp.at<float>(i, 2 * j) - imgtmp.at<float>(i, 2 * j + 1)) / 2;
            }
        }
        for (int i = 0; i < Width / 2; i++){
            for (int j = 0; j < Height; j++){
                wavelet.at<float>(i, j) = (tmp.at<float>(2 * i, j) + tmp.at<float>(2 * i + 1, j)) / 2;
                wavelet.at<float>(i + Width / 2, j) = (tmp.at<float>(2 * i, j) - tmp.at<float>(2 * i + 1, j)) / 2;
            }
        }
        imgtmp = wavelet;
        depthcount++;
    }

    wavelet.convertTo(wavelet, CV_8UC1);
    imwrite("output.png", wavelet);
    return 0;
}
