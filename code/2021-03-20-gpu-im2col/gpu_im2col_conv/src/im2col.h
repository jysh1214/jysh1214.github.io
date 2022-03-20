#ifndef IM_TO_COL_H
#define IM_TO_COL_H

#include <cuda.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp> // cv::cuda::PtrStepSz
#include <opencv2/cudaarithm.hpp>
#include <iostream>

extern "C" void gpu_im2col(cv::Mat& src, int k_rows, int k_cols, int padding, int stride, cv::Mat& dst);
extern "C" void gpu_conv(const cv::Mat& src, const cv::Mat& kernel, const int padding, const int stride, cv::Mat& dst);

#endif