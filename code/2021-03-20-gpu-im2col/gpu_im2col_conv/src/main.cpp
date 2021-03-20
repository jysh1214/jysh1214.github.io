#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#include "im2col.h"

int main()
{
    float src_data[4][4] = {
        { 0, 1, 2, 3},
        { 4, 5, 6, 7},
        { 8, 9,10,11},
        {12,13,14,15}
    };
    cv::Mat src = cv::Mat(4, 4, CV_32F, src_data);

    // float kernel_data[2][2] = {
    //     {2, 1},
    //     {2, 2}
    // };
    // cv::Mat kernel = cv::Mat(2, 2, CV_32F, kernel_data);

    float kernel_data[3][3] = {
        {2, 2, 1},
        {2, 2, 1},
        {1, 1, 1}
    };
    cv::Mat kernel = cv::Mat(3, 3, CV_32F, kernel_data);

    cv::Mat conv_result;
    gpu_conv(src, kernel, 1, 1, conv_result);

    std::cout << conv_result << "\n";

    return 0;
}
