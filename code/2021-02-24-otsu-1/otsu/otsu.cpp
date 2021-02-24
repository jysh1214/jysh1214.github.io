#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include "merge.hpp"

// clang++ -std=c++11 otsu.cpp `pkg-config --libs opencv4` `pkg-config --cflags opencv4` -lstdc++

int main()
{
    cv::Mat src, gray;
    src = cv::imread("test.jpg");
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    cv::Mat dst;
    double thresholdValue = cv::threshold(gray, dst, 0, 255, cv::THRESH_OTSU);
    // cv::imwrite("output.png", dst);

    std::cout << thresholdValue << "\n";

    cv::Mat final;
    merge(gray, dst, final);
    cv::imwrite("output.png", final);

    return 0;
}
