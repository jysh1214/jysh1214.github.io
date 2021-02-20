#include <opencv2/core.hpp>
#include <iostream>

// clang++ gemm.cpp -std=c++11 `pkg-config --libs opencv4` `pkg-config --cflags opencv4` -lstdc++

int main()
{
    float arr_a[3][3] = {
        {2, 2, 2},
        {1, 1, 1},
        {1, 1, 1}
    };
    float arr_b[3][3] = {
        {1, 1, 1},
        {2, 1, 1},
        {1, 1, 1}
    };
    cv::Mat matrix_a{3, 3, CV_32F, arr_a};
    cv::Mat matrix_b{3, 3, CV_32F, arr_b};

    cv::Mat matrix_c;
    cv::gemm(matrix_a, matrix_b, 1.0, cv::Mat(), 0.0, matrix_c);
    std::cout << matrix_c << "\n";

    return 0;
}
