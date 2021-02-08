#include "merge.hpp"
#include "usan.hpp"
#include "sobel.hpp"
#include "canny.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main(int argc, char** argv)
{
    cv::Mat src = cv::imread("test.jpg");
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    cv::Mat c;
    canny(gray, c);

    cv::Mat output;
    cv::cvtColor(c, c, cv::COLOR_GRAY2RGB);
    merge(src, c, output);
    cv::imwrite("output.png", output);

    return 0;
};
