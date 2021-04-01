#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

// clang++ main.cpp -std=c++11 `pkg-config --libs opencv4` `pkg-config --cflags opencv4` -lstdc++

int main()
{
    cv::Mat src = cv::imread("origin.jpg");
    cv::Mat ycrcb;

    cv::cvtColor(src, ycrcb, cv::COLOR_RGB2YCrCb);
    std::vector<cv::Mat> channels;
    cv::split(ycrcb, channels);

    cv::equalizeHist(channels[0], channels[0]);
    cv::Mat dst;
    cv::merge(channels, ycrcb);
    cv::cvtColor(ycrcb, dst, cv::COLOR_YCrCb2RGB);
    cv::imwrite("output.png", dst);

    return 0;
}
