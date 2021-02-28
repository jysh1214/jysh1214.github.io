---
layout: post
title:  "Haar 小波轉換"
---

小波轉換，是一種時域轉頻域的數學工具，相較於傅立葉轉換基底為弦波，小波轉換的基底能量集中在有限時間。
小波轉換對於局部劇烈變化，較傅立葉轉換來得有效率。

對於電腦影像因為是離散數據，則是使用離散小波轉換`(Discrete Wavelet Transform, DWT)`。
本文使用`離散 Haar 小波轉換`來找出圖片中局部變化較大的部分。

離散 Haar 小波轉換分為兩個步驟：
1. 計算均值： $$ avg_m = (x_{2m} + x_{2m+1}) / 2 $$
2. 計算差值： $$ diff_m = (x_{2m} - x_{2m+1}) / 2 $$

example:
$$
x = [1, 3, 5, 7] \\
avg = [(1+3)/2, (5+7)/2] = [2, 6] \\
diff = [(1-3)/2, (5-7)/2] = [-1, -1]
$$

則數列 $$ x $$ 經過一次`離散 Haar 小波轉換`為 $$ [2, 6, -1, -1] $$。

二維離散 Haar 小波轉換需要行列各做一次：
- 行小波分解
- 列小波分解

上述步驟可以互換，不影響結果。

## Implement:
為更好瞭解`離散 Haar 小波轉換`，先用 $$ 4 \times 4 $$ 矩陣來實作。

```bash=
原始矩陣：
[0, 2, 4, 6;
 8, 10, 12, 14;
 16, 18, 20, 22;
 24, 26, 28, 30]
行小波分解：
[1, 5, -1, -1;
 9, 13, -1, -1;
 17, 21, -1, -1;
 25, 29, -1, -1]
列小波分解：
[5, 9, -1, -1;
 21, 25, -1, -1;
 -4, -4, 0, 0;
 -4, -4, 0, 0]
結果：
[5, 9, -1, -1;
 21, 25, -1, -1;
 -4, -4, 0, 0;
 -4, -4, 0, 0]
```
將步驟互換，先`列小波分解`再`行小波分解`：
```bash=
原始矩陣：
[0, 2, 4, 6;
 8, 10, 12, 14;
 16, 18, 20, 22;
 24, 26, 28, 30]
列小波分解：
[4, 6, 8, 10;
 20, 22, 24, 26;
 -4, -4, -4, -4;
 -4, -4, -4, -4]
行小波分解：
[5, 9, -1, -1;
 21, 25, -1, -1;
 -4, -4, 0, 0;
 -4, -4, 0, 0]
結果：
[5, 9, -1, -1;
 21, 25, -1, -1;
 -4, -4, 0, 0;
 -4, -4, 0, 0]
```
對結果沒影響。

對圖片實作二階轉換（做兩次轉換）：
```c++=
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

int main() {
    Mat img = imread("test.jpg", 0);
    int Height = img.cols;
    int Width = img.rows;

    int depth = 2;
    int depthcount = 1;
    Mat tmp = Mat::ones(Width, Height, CV_32FC1);
    Mat wavelet = Mat::ones(Width, Height, CV_32FC1);
    Mat imgtmp = img.clone();
    imgtmp.convertTo(imgtmp, CV_32FC1);
    while (depthcount<=depth){
        Width = img.rows / depthcount;
        Height = img.cols / depthcount;

        for (int i = 0; i < Width; i++){
            for (int j = 0; j < Height / 2; j++){
                tmp.at<float>(i, j) = (imgtmp.at<float>(i, 2 * j) + imgtmp.at<float>(i, 2 * j + 1)) / 2;
                tmp.at<float>(i, j + Height / 2) = (imgtmp.at<float>(i, 2 * j) - imgtmp.at<float>(i, 2 * j + 1)) / 2;
            }
        }
        for (int i = 0; i < Width / 2; i++){
            for (int j = 0; j < Height; j++){
                wavelet.at<float>(i, j) = (tmp.at<float>(2 * i, j) + tmp.at<float>(2 * i + 1, j)) / 2;
                wavelet.at<float>(i + Width / 2, j) = (tmp.at<float>(2 * i, j) - tmp.at<float>(2 * i + 1, j)) / 2;
            }
        }
        imgtmp = wavelet;
        depthcount++;
    }

    wavelet.convertTo(wavelet, CV_8UC1);
    imwrite("output.png", wavelet);
    return 0;
}
```
## Result:

<img src="/assets/images/2021-02-28-wavelet-haar/output.png" width="1300">

## References:

1. [小波十講](https://www.books.com.tw/products/CN11408747)
2. [圖片馬賽克的建置](http://www.cs.thu.edu.tw/upload_files/96_cttsai_01.pdf)
3. [opencv小练习：哈尔小波(Haar)](https://blog.csdn.net/u010006643/article/details/50493566)
4. [小波变换完美通俗讲解](https://zhuanlan.zhihu.com/p/44215123)
