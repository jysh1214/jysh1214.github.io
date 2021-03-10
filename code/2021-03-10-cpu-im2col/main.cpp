#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// clang++ main.cpp -std=c++11 `pkg-config --libs opencv4` `pkg-config --cflags opencv4` -lstdc++

cv::Mat cpu_im2col(cv::Mat& src, int k_rows, int k_cols, int padding, int stride)
{
    int dst_rows = (src.rows + 2 * padding - k_rows) / stride + 1;
    int dst_cols = (src.cols + 2 * padding - k_cols) / stride + 1;

    cv::Mat padded = cv::Mat::zeros(src.rows + 2 * padding, src.cols + 2 * padding, CV_32F);
    src.copyTo(padded(cv::Rect(padding, padding, src.cols, src.rows)));

    cv::Mat dst(dst_rows * dst_cols, k_rows * k_cols, CV_32F);

    int current_row = 0;
    for (int i = 0; i < padded.rows; i += stride) {
        for (int j = 0; j < padded.cols; j+= stride) {
            if (i + k_rows > padded.rows || j + k_cols > padded.cols)
                continue;

            for (int r = 0; r < k_rows; r++) {
                for (int c = 0; c < k_cols; c++) {
                    int src_row = i - (k_rows - 1) / 2 + r;
                    int src_col = j - (k_cols - 1) / 2 + c;
                    dst.at<float>(current_row, r * k_cols + c) = padded.at<float>(src_row, src_col);
                }
            }
            current_row++;
        }
    }

    return dst;
}

int main()
{
    cv::Mat src, im2col_conv, filter_conv;
    float src_data[4][4] = {
        { 0,  1,  2,  3},
        { 4,  5,  6,  7},
        { 8,  9, 10, 11},
        {12, 13, 14, 15}
    };
    src = cv::Mat(4, 4, CV_32F, src_data);
    im2col_conv = cpu_im2col(src, 2, 2, 1, 1);

    // std::cout << im2col_conv << "\n";

    cv::Mat kernel;
    float kernel_data[2][2] = {
        {2, 2},
        {2, 2}
    };
    kernel = cv::Mat(2, 2, CV_32F, kernel_data);

    cv::filter2D(src, filter_conv, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);

    // std::cout << filter_conv << "\n";


    return 0;
}
