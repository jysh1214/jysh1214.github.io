#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// clang++ main.cpp -std=c++11 `pkg-config --libs opencv4` `pkg-config --cflags opencv4` -lstdc++

cv::Mat cpu_im2col(const cv::Mat& src, int k_rows, int k_cols, int padding, int stride)
{
    int dst_rows = (src.rows + 2 * padding - k_rows) / stride + 1;
    int dst_cols = (src.cols + 2 * padding - k_cols) / stride + 1;

    cv::Mat padded = cv::Mat::zeros(src.rows + 2 * padding, src.cols + 2 * padding, CV_32F);
    src.copyTo(padded(cv::Rect(padding, padding, src.cols, src.rows)));

    cv::Mat dst(dst_rows * dst_cols, k_rows * k_cols, CV_32F);

    int k_rows_radius = k_rows / 2;
    int k_cols_radius = k_cols / 2;
    int current_row = 0;
    for (int i = k_rows_radius; i < padded.rows - k_rows_radius; i += stride) {
        for (int j = k_cols_radius; j < padded.cols - k_cols_radius; j+= stride) {

            for (int r = 0; r < k_rows; r++) {
                for (int c = 0; c < k_cols; c++) {
                    int src_row = i - k_rows_radius+ r;
                    int src_col = j - k_cols_radius + c;
                    dst.at<float>(current_row, r * k_cols + c) = padded.at<float>(src_row, src_col);
                }
            }
            current_row++;
        }
    }

    return dst;
}

cv::Mat conv(const cv::Mat& src, const cv::Mat& kernel, int padding, int stride)
{
    assert(kernel.rows <= src.rows && kernel.cols <= src.cols);

    int k_rows = kernel.rows;
    int k_cols = kernel.cols;

    cv::Mat reshape_src = cpu_im2col(src, k_rows, k_cols, padding, stride);
    std::cout << reshape_src << "\n";
    cv::Mat reshape_ker = kernel.reshape(kernel.channels(), kernel.rows * kernel.cols).clone();
    cv::Mat empty, gemm_result;
    cv::gemm(reshape_src, reshape_ker, 1.0, empty, 0.0, gemm_result);

    int dst_rows = (src.rows + 2 * padding - k_rows) / stride + 1;
    cv::Mat result = gemm_result.reshape(gemm_result.channels(), dst_rows).clone();

    return result;
}

int main()
{
    cv::Mat src, im2col_mat;
    float src_data[4][4] = {
        { 0,  1,  2,  3},
        { 4,  5,  6,  7},
        { 8,  9, 10, 11},
        {12, 13, 14, 15}
    };
    src = cv::Mat(4, 4, CV_32F, src_data);
    cv::Mat kernel, new_kernel;
    float kernel_data[3][3] = {
        {2, 2, 1},
        {2, 2, 1},
        {1, 1, 1}
    };
    kernel = cv::Mat(3, 3, CV_32F, kernel_data);

    cv::Mat conv_result = conv(src, kernel, 1, 1);

    std::cout << conv_result << "\n";

    cv::Mat filter_conv;
    cv::filter2D(src, filter_conv, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
    std::cout << filter_conv << "\n";


    return 0;
}
