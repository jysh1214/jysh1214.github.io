#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_device_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <assert.h>

int main()
{
    cv::Mat src, filter, dst;
    cv::cuda::GpuMat cu_src, cu_gray, cu_filter, cu_dst;

    int filter_height = 3;
    int filter_width = 3;

    float filter_data[filter_height][filter_width] = {
        {-1, -1, -1},
        {-1,  8, -1},
        {-1, -1, -1}
    };

    filter = cv::Mat(filter_height, filter_width, CV_32FC1, filter_data);
    cu_filter.upload(filter);

    cv::Ptr<cv::cuda::Convolution> conv = cv::cuda::createConvolution(cv::Size(filter_height, filter_width));

    src = cv::imread("test.jpg");
    cu_src.upload(src);
    cv::cuda::cvtColor(cu_src, cu_gray, cv::COLOR_BGR2GRAY);

    // cuda convolve: Only CV_32FC1 images are supported for now. 
    cu_gray.convertTo(cu_gray, CV_32FC1);
    

    conv->convolve(cu_gray, cu_filter, cu_dst);
    cu_dst.download(dst);

    dst.convertTo(dst, CV_8UC1);
    cv::imwrite("gpu_output.png", dst);

    // cpu
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::filter2D(gray, dst, -1, filter, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
    cv::imwrite("cpu_output.png", dst);

    return 0;
}