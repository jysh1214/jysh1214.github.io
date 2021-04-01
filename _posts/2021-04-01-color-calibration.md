---
layout: post
title:  "Color Calibration 演算法（一）"
---

使用高品質的圖像可以讓影像處理效果更佳，但受限於~~使用者~~環境或是設備影響，
圖像的品質可能不盡理想。

對於光線造成圖像整體過於明亮或是黯淡，可以使用`色彩校正(Color Calibration)`進行處理。

以下為常見的演算法：
- Histogram Equalization (HE)
- Exact Histogram Specification (EHS)
- Layered Difference Representation (LDR)
- Histogram Modification Framework (HMF)
- Recursively Separated and Weighted Histogram Equalization (RSWHE)
- Adaptive Gamma Correction with Weighting Distribution (AGCWD)

## Histogram Equalization (HE)

原始圖像的亮度可能集中在某一區間，使用`HE`讓亮度更平均分佈，達到校正效果。

令原始影像在座標 $$(x, y)$$ 的亮度為 $$x_{x, y}$$，則調整過後亮度為

$$
h(x_{x, y}) = round( \frac{cdf(x_{x, y}) - cdf_{min}}{cdf_{max} - cdf_{min}} * (L - 1) )
$$

其中，$$ cdf $$為各亮度強度出現機率的累積分佈函數，為

$$
\sum p(x) = \sum \frac{hist(x)}{n}
$$

其中$$ hist(x) $$為亮度$$ x $$出現個數，$$ n $$為總像素個數。

`HE`是將圖像亮度做平均，對於色彩圖片，直接對`RGB`分量做`HE`，會導致失真，
要先將彩色圖片轉換到以下三種色域空間之一，再對亮度做均衡。
- HSV/HLS
- YUV
- YCbCr

使用`OpenCV`內建函式實作。
```c++=
int main()
{
    cv::Mat src = cv::imread("origin.jpg");
    cv::Mat ycrcb;
    // 轉到 YCbCr 色域空間
    cv::cvtColor(src, ycrcb, cv::COLOR_RGB2YCrCb);
    std::vector<cv::Mat> channels;
    cv::split(ycrcb, channels);
    // 單獨對亮度做均衡化
    cv::equalizeHist(channels[0], channels[0]);
    cv::Mat dst;
    // 亮度均衡化後合併
    cv::merge(channels, ycrcb);
    cv::cvtColor(ycrcb, dst, cv::COLOR_YCrCb2RGB);
    cv::imwrite("output.png", dst);

    return 0;
}
```

原始圖像：
<center><img src="/assets/images/2021-04-01-color-calibration/origin.jpg" width="650"></center>

原始圖像直方圖：
<center><img src="/assets/images/2021-04-01-color-calibration/origin_plot.png" width="650"></center>

結果圖像：
<center><img src="/assets/images/2021-04-01-color-calibration/output.png" width="650"></center>

結果圖像直方圖：
<center><img src="/assets/images/2021-04-01-color-calibration/output_plot.png" width="650"></center>

可以看出`HE`雖然簡單好用，但不是對所有圖效果都很好。

## References
- [Histogram equalization not working on color image - OpenCV
](https://stackoverflow.com/questions/15007304/histogram-equalization-not-working-on-color-image-opencv)
- [Photo by Илья Косарев on Unsplash](https://unsplash.com/@ikocarev?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)
