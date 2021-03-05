#ifndef ENTROPY_HPP
#define ENTROPY_HPP

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <assert.h>
#include <math.h>

#define ASSIGN(arr_a, arr_b, current_size, arr_size) \
do {                                                 \
    int offset = current_size - arr_size;            \
    for (int i = 0; i < arr_size; i++)               \
        arr_a[i+offset] = arr_b[i];                  \
} while (0)

int get_size(int radius)
{
    int size = radius * 2 + 1;
    for (int i = 1, j = size; i < radius; ++i)
        size += (j = j-2);
    size = size * 2 + (radius * 2 + 1);

    return size;
}

/**
 * @return: [-(radius), -(radius-1), ..., 0, ..., radius-1, radius]
 */
int* mirror_arr(int radius)
{
    assert(radius > 0);

    int size = radius * 2 + 1;
    int* arr = new int[size];
    for (int val = -radius, index = 0; index < size; index++) {
        arr[index] = val++;
    }

    return arr;
}

int* create_offset_x(int radius)
{
    assert(radius > 0);

    int size = get_size(radius);
    int* offset_x = new int[size];
    int current_size = 0;
    for (int i = 0; i < (2*radius+1); i++) {
        int* arr = nullptr;
        int arr_size = 0;
        if (i == radius) {
            arr = mirror_arr(i);
            arr_size = i * 2 + 1;
        }
        else if (i < radius) {
            arr = mirror_arr(i + 1);
            arr_size = (i + 1) * 2 + 1;
        }
        else if (i > radius) {
            int diff = i - radius - 1;
            arr = mirror_arr(radius - diff);
            arr_size = (radius - diff) * 2 + 1;
        }
        current_size += arr_size;
        ASSIGN(offset_x, arr, current_size, arr_size);
        if (arr) delete arr;
    }

    return offset_x;
}

int* create_offset_y(int radius)
{
    assert(radius > 0);

    int size = get_size(radius);
    int* offset_y = new int[size];
    int current_size = 0;
    int val = -radius;
    current_size = 0;
    for (int j = 0; j < (2*radius+1); j++) {
        int* arr = nullptr;
        int arr_size = 0;
        if (j == radius) {
            arr_size = j * 2 + 1;
            arr = new int[arr_size];
        }
        else if (j < radius) {
            arr_size = (j + 1) * 2 + 1;
        }
        else if (j > radius) {
            int diff = j - radius - 1;
            arr_size = (radius - diff) * 2 + 1;
        }
        current_size += arr_size;
        arr = new int[arr_size];
        for (int a = 0; a < arr_size; a++)
            arr[a] = val;
        val++;
        ASSIGN(offset_y, arr, current_size, arr_size);
        if (arr) delete arr;
    }

    return offset_y;
}

static
void energy(const cv::Mat& src, cv::Mat& dst, int radius, int mode)
{
    assert(src.type() == CV_8UC1);
    assert(radius >= 1);
    assert(mode == 0 || mode == 1 || mode == 2);

    dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);

    int* offset_x = create_offset_x(radius);
    int* offset_y = create_offset_y(radius);

    assert(offset_x && offset_y);

    // scan
    int size = get_size(radius);
    for (int x = radius; x < src.rows - radius; x++) {
        for (int y = radius; y < src.cols - radius; y++) {
            int sum = 0;
            if (mode == 0) { // circle mask
                for (int offset = 0; offset < size; offset++) {
                    sum += src.at<uchar>(x + offset_x[offset], y + offset_y[offset]);
                }
            }
            else if (mode == 1) { // square mask
                for (int a = x - radius; a < x + radius; a++) {
                    for (int b = y - radius; b < y + radius; b++) {
                        sum += src.at<uchar>(a, b);
                    }
                }
            }
            else if (mode == 2) { // 左右橫向
                for (int b = y - radius; b < y + radius; b++)
                    sum += src.at<uchar>(x, b);
            }

            dst.at<uchar>(x, y) = sum;
        }
    }

    if (offset_x) delete offset_x;
    if (offset_y) delete offset_y;
}

static
void entropy(const cv::Mat& src, cv::Mat& dst, int radius, int mode)
{
    assert(src.type() == CV_8UC1);
    assert(radius >= 1);

    cv::Mat energy_map;
    energy(src, energy_map, radius, mode);
    for (int x = 0; x < energy_map.rows; x++) {
        for (int y = 0; y < energy_map.cols; y++) {
            uchar energy = energy_map.at<uchar>(x, y);
            energy_map.at<uchar>(x, y) = energy * log(energy);
        }
    }
    dst = energy_map.clone();
}

#endif
