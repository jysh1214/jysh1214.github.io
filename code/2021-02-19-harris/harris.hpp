#ifndef DRAW_CORNER_HPP
#define DRAW_CORNER_HPP

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

/**
 * @src: 黑白圖
 * @dst: 角落圖上灰色的圓
 * @threshold: Harris角點響應值大於此值為角點
 */
 void drawCorner(const cv::Mat& src, cv::Mat& dst, int threshold)
 {
     assert(src.type() == CV_8U);
     int blockSize = 5;
     int apertureSize = 3;
     double k = 0.04;
     cv::Mat harris;
     // 回傳與輸入圖片大小一樣的 mat，其值為 Harris 角點響應
     cv::cornerHarris(src, harris, blockSize, apertureSize, k);

     cv::Mat draw = src.clone();
     cv::Mat harris_norm, harris_norm_scaled;
     cv::normalize(harris, harris_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
     cv::convertScaleAbs(harris_norm, harris_norm_scaled);

     for (int i = 0; i < harris_norm.rows; i++) {
         for (int j = 0; j < harris_norm.cols; j++) {
             if ((int)harris_norm_scaled.at<uchar>(i, j) > threshold) {
                 cv::circle(draw, cv::Point(j, i), 30, cv::Scalar(125), 2, 8, 0);
             }
         }
     }

     dst = draw.clone();
 }

#endif
