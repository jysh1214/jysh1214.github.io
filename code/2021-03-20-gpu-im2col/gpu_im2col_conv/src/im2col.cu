#include "im2col.h"

__global__ void im2col_kernel(cv::cuda::PtrStepSz<float1> cu_padded, 
    int k_rows,
    int k_cols,
    int stride,
    cv::cuda::PtrStepSz<float1> cu_dst)
{
    unsigned int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int id_y = blockIdx.y * blockDim.y + threadIdx.y;

    int k_rows_radius = k_rows / 2;
    int k_cols_radius = k_cols / 2;

    unsigned int i = k_rows_radius + id_x * stride;
    unsigned int j = k_cols_radius + id_y * stride;

    int rows_limit = (k_rows % 2 == 0)? (cu_padded.rows - k_rows_radius + 1): (cu_padded.rows - k_rows_radius);
    int cols_limit = (k_cols % 2 == 0)? (cu_padded.cols - k_cols_radius + 1): (cu_padded.cols - k_cols_radius);

    if (i < rows_limit && j < cols_limit) {
        unsigned int current_row = id_x * ((cols_limit - k_cols_radius) / stride) + id_y;
        for (int r = 0; r < k_rows; r++) {
            for (int c = 0; c < k_cols; c++) {
                int src_row = i - k_rows_radius + r;
                int src_col = j - k_cols_radius + c;
                cu_dst(current_row, r * k_cols + c) = cu_padded(src_row, src_col);
            }
        }
    }
}

void gpu_im2col(const cv::Mat& src, const int k_rows, const int k_cols, const int padding, const int stride, cv::Mat& dst)
{
    assert(src.rows >= k_rows && src.cols >= k_cols);
    assert(padding >= 0 && stride >= 1);

    int conv_rows = (src.rows + 2 * padding - k_rows) / stride + 1;
    int conv_cols = (src.cols + 2 * padding - k_cols) / stride + 1;

    cv::Mat padded = cv::Mat::zeros(src.rows + 2 * padding, src.cols + 2 * padding, CV_32F);
    src.copyTo(padded(cv::Rect(padding, padding, src.cols, src.rows)));

    cv::cuda::GpuMat cu_padded;
    cu_padded.upload(padded);

    int blockSize = 8;

    int grid_x = (conv_rows + blockSize - 1) / blockSize; // src rows
    int grid_y = (conv_cols + blockSize - 1) / blockSize; // src cols

    dim3 grid(grid_x, grid_y, 1);
    dim3 block(blockSize, blockSize, 1);

    cv::cuda::GpuMat cu_dst;
    dst = cv::Mat(conv_rows * conv_cols, k_rows * k_cols, CV_32F, cv::Scalar(0.0));
    cu_dst.upload(dst);

    im2col_kernel<<<grid, block>>>(cu_padded, k_rows, k_cols, stride, cu_dst);
    
    if (cudaSuccess != cudaGetLastError()) {
        std::cout << "kernel fault.\n";
    }

    cu_dst.download(dst);
}

void gpu_conv(const cv::Mat& src, const cv::Mat& kernel, const int padding, const int stride, cv::Mat& dst)
{
    assert(kernel.rows <= src.rows && kernel.cols <= src.cols);

    int k_rows = kernel.rows;
    int k_cols = kernel.cols;

    cv::Mat reshape_src;
    gpu_im2col(src, k_rows, k_cols, padding, stride, reshape_src);

    cv::Mat reshape_ker = kernel.reshape(kernel.channels(), kernel.rows * kernel.cols).clone();
    cv::Mat empty, gemm_result;
    cv::cuda::gemm(reshape_src, reshape_ker, 1.0, empty, 0.0, gemm_result);
    
    int dst_rows = (src.rows + 2 * padding - k_rows) / stride + 1;
    dst = gemm_result.reshape(gemm_result.channels(), dst_rows).clone();
}