#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include "entropy.hpp"

// clang++ main.cpp -std=c++11 `pkg-config --libs opencv4` `pkg-config --cflags opencv4` -lstdc++

int main() {
    cv::Mat img = cv::imread("test.jpg", 0);
    int width = img.rows;
    int height = img.cols;

    int level = 2;
    int decompose = 1;
    cv::Mat tmp = cv::Mat::ones(width, height, CV_32FC1);
    cv::Mat wav = cv::Mat::ones(width, height, CV_32FC1);
    cv::Mat imgtmp = img.clone();
    imgtmp.convertTo(imgtmp, CV_32FC1);
    while (decompose <= level) {
        width = img.rows / decompose;
        height = img.cols / decompose;

        for (int i = 0; i < width; i++){
            for (int j = 0; j < height / 2; j++){
                tmp.at<float>(i, j) = (imgtmp.at<float>(i, 2 * j) + imgtmp.at<float>(i, 2 * j + 1)) / 2;
                tmp.at<float>(i, j + height / 2) = (imgtmp.at<float>(i, 2 * j) - imgtmp.at<float>(i, 2 * j + 1)) / 2;
            }
        }
        for (int i = 0; i < width / 2; i++){
            for (int j = 0; j < height; j++){
                wav.at<float>(i, j) = (tmp.at<float>(2 * i, j) + tmp.at<float>(2 * i + 1, j)) / 2;
                wav.at<float>(i + width / 2, j) = (tmp.at<float>(2 * i, j) - tmp.at<float>(2 * i + 1, j)) / 2;
            }
        }
        imgtmp = wav;
        decompose++;
    }

    wav.convertTo(wav, CV_8UC1);
    // imwrite("output.png", wav);

    cv::Rect roi(img.cols/2, img.rows/2, img.cols/2, img.rows/2);
    cv::Mat HH = wav(roi);

    int mode = 1;
    cv::Mat energy_map;
    entropy(HH, energy_map, 3, mode);
    imwrite("entropy.png", energy_map);

    return 0;
}
