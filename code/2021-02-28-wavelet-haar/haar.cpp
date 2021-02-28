#include <opencv2/opencv.hpp>
#include <iostream>

// clang++ haar.cpp -std=c++11 `pkg-config --libs opencv4` `pkg-config --cflags opencv4` -lstdc++

using namespace std;
using namespace cv;

int main()
{
    float arrdata[4][4];

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            arrdata[i][j] = (i * 4 + j) * 2;
        }
    }

    Mat img{4, 4, CV_32F, arrdata};

    cout << "img: " << "\n";
    cout << img << "\n";

    int Height = img.cols;
    int Width = img.rows;

    int depth = 1;
    int depthcount = 1;
    Mat tmp = Mat::ones(Width, Height, CV_32FC1);
    Mat wavelet = Mat::ones(Width, Height, CV_32FC1);
    Mat imgtmp = img.clone();

    imgtmp.convertTo(imgtmp, CV_32FC1);

    while (depthcount <= depth) {
        Width = img.rows / depthcount;
        Height = img.cols / depthcount;

        // cout << "列小波分解： " << "\n";
        //
        // for (int i = 0; i < Width / 2; i++){
        //     for (int j = 0; j < Height; j++){
        //         tmp.at<float>(i, j) = (imgtmp.at<float>(2 * i, j) + imgtmp.at<float>(2 * i + 1, j)) / 2;
        //         tmp.at<float>(i + Width / 2, j) = (imgtmp.at<float>(2 * i, j) - imgtmp.at<float>(2 * i + 1, j)) / 2;
        //     }
        // }
        //
        // cout << tmp << "\n";
        //
        // cout << "行小波分解： " << "\n";
        //
        // for (int i = 0; i < Width; i++){
        //     for (int j = 0; j < Height / 2; j++){
        //         wavelet.at<float>(i, j) = (tmp.at<float>(i, 2 * j) + tmp.at<float>(i, 2 * j + 1)) / 2;
        //         wavelet.at<float>(i, j + Height / 2) = (tmp.at<float>(i, 2 * j) - tmp.at<float>(i, 2 * j + 1)) / 2;
        //     }
        // }

        cout << "行小波分解： " << "\n";

        for (int i = 0; i < Width; i++){
            for (int j = 0; j < Height / 2; j++){
                tmp.at<float>(i, j) = (imgtmp.at<float>(i, 2 * j) + imgtmp.at<float>(i, 2 * j + 1)) / 2;
                tmp.at<float>(i, j + Height / 2) = (imgtmp.at<float>(i, 2 * j) - imgtmp.at<float>(i, 2 * j + 1)) / 2;
            }
        }

        cout << tmp << "\n";

        cout << "列小波分解： " << "\n";

        for (int i = 0; i < Width / 2; i++){
            for (int j = 0; j < Height; j++){
                wavelet.at<float>(i, j) = (tmp.at<float>(2 * i, j) + tmp.at<float>(2 * i + 1, j)) / 2;
                wavelet.at<float>(i + Width / 2, j) = (tmp.at<float>(2 * i, j) - tmp.at<float>(2 * i + 1, j)) / 2;
            }
        }

        cout << wavelet << "\n";

        imgtmp = wavelet;
        depthcount++;
    }

    cout << "final: " << "\n";
    cout << wavelet << "\n";




    return 0;
}
