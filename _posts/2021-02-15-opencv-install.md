---
layout: post
title:  "OpenCV Install with CUDA"
---

Install OpenCV with
- CUDA
- CUDNN
- pkg config
- gstreamer

```bash=
mkdir opencv_install
cd opencv_install
```

```bash=
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
```

```bash=
mkdir build
cd build
```

```bash=
sudo apt install libopencv-dev
```

```bash=
cmake -D CMAKE_BUILD_TYPE=Release \
-D OPENCV_GENERATE_PKGCONFIG=YES \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D WITH_TBB=ON \
-D WITH_V4L=ON \
-D INSTALL_C_EXAMPLES=ON \
-D WITH_CUBLAS=1 \
-D WITH_LIBV4L=ON \
-D WITH_GSTREAMER=ON \
-D WITH_GSTREAMER_0_10=OFF \
-D WITH_NVCUVID=ON \
-D FORCE_VTK=ON \
-D WITH_XINE=ON \
-D WITH_CUDA=ON \
-D WITH_CUDNN=ON \
-D CUDNN_INCLUDE_DIR=/usr/local/cuda/include/ \
-D CUDNN_LIBRARY=/lib/x86_64-linux-gnu/libcudnn.so.8 \
-D CUDNN_VERSION=8.0.4 \
-D CUDA_ARCH_BIN=11.1 \
-D CUDA_GENERATION=Pascal \
-D OPENCV_DNN_CUDA=ON \
-D CUDA_NVCC_FLAGS="-D_FORCE_INLINES --expt-relaxed-constexpr" \
-D ENABLE_FAST_MATH=1 \
-D CUDA_FAST_MATH=1 \
-D WITH_GDAL=ON \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D PYTHON_EXECUTABLE=/usr/bin/python3 \
-D OPENCV_PYTHON3_INSTALL_PATH=/usr/local/lib/python3.8/dist-packages \
-DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \
 ../opencv
```
其中，要注意`CUDA`和`python`路徑要改成自己的環境。

```bash=
make
```
```bash=
sudo make install
```

## MacOS cmake options
```bash=
cmake -D CMAKE_BUILD_TYPE=Release \
-D OPENCV_GENERATE_PKGCONFIG=YES \
-D INSTALL_C_EXAMPLES=ON \
-D FORCE_VTK=ON \
-D WITH_XINE=ON \
-D ENABLE_FAST_MATH=1 \
-D WITH_GDAL=ON \
-D BUILD_NEW_PYTHON_SUPPORT=ON \
-D BUILD_opencv_python3=ON \
-D HAVE_opencv_python3=ON \
-D PYTHON_DEFAULT_EXECUTABLE=/usr/bin/python3 \
-DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \
 ../opencv
```
