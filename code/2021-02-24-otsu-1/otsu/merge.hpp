#ifndef MERGE_HPP
#define MERGE_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <assert.h>

/**
 * 合併兩張圖 左: input1 右: input2
 */
static void merge(const cv::Mat& input1, const cv::Mat& input2, cv::Mat& output)
{
    assert(input1.rows == input2.rows && input1.cols == input2.cols);
    assert(input1.type() == input2.type());

    output = cv::Mat(input1.rows, input1.cols*2, input1.type());

    if (output.channels() == 1) {
        for (int i = 0; i < output.rows; ++i) {
            for (int j = 0; j < output.cols; ++j) {
                output.at<uchar>(i, j) = (j < input1.cols)? input1.at<uchar>(i, j): input2.at<uchar>(i, j);
            }
        }
    }
    else {
        for (int i = 0; i < output.rows; ++i) {
            for (int j = 0; j < output.cols; ++j) {
                for (int c = 0; c < output.channels(); ++c) {
                    output.at<cv::Vec3b>(i, j)[c] = (j < input1.cols)? input1.at<cv::Vec3b>(i, j)[c]:
                                                         input2.at<cv::Vec3b>(i, j)[c];
                }
            }
        }
    }
}

#endif
