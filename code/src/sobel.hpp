#ifndef SOBEL_HPP
#define SOBEL_HPP

void sobel(const cv::Mat& src, cv::Mat& dst, int ksize, int scale, int delta)
{
    int ddepth = -1;
    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y;
    cv::Sobel(src, grad_x, ddepth, 1, 0, ksize, scale, delta, cv::BORDER_DEFAULT);
    cv::Sobel(src, grad_y, ddepth, 0, 1, ksize, scale, delta, cv::BORDER_DEFAULT);

    // converting back to CV_8U
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::convertScaleAbs(grad_y, abs_grad_y);

    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);
}

#endif
