#ifndef CANNY_HPP
#define CANNY_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

static void canny(const cv::Mat& src, cv::Mat& dst)
{
    cv::Canny(src, dst, 50, 150);
}

#endif
