#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

// clang++ -std=c++11 probability.cpp `pkg-config --libs opencv4` `pkg-config --cflags opencv4` -lstdc++

void getHistValue(const cv::Mat& src, std::vector<float>& histValue)
{
    const int channels[1] = {0};
    const int histSize[1] = {256};
    float hrange[2] = {0, 255};
    const float* range[1] = {hrange};
    cv::MatND histMat;
    cv::calcHist(&src, 1, channels, cv::Mat(), histMat, 1, histSize, range);

    for (int i = 0; i < 256; i++)
        histValue.push_back(histMat.at<float>(i));
}

int main()
{
    cv::Mat src, gray;
    src = cv::imread("test.jpg");
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    float total = gray.rows * gray.cols;
    std::vector<float> histValue;
    getHistValue(gray, histValue);

    std::cout << "Gray Levels, " << " Probability\n";

    for (int i = 0; i < 256; i++)
        std::cout << i << ", " << histValue[i]/total << "\n";

    return 0;
}
