#include "harris.hpp"
#include "merge.hpp"

// clang++ main.cpp -std=c++11 `pkg-config --libs opencv4` `pkg-config --cflags opencv4` -lstdc++

int main()
{
    cv::Mat src;
    src = cv::imread("test.png");
    cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);

    cv::Mat dst;
    drawCorner(src, dst, 200);

    // cv::imwrite("output.png", dst);
    cv::Mat mm;
    merge(src, dst, mm);
    cv::imwrite("output.png", mm);

    return 0;
}
