#include <stdio.h>
#include <cuda.h>

__global__ void dkernel(unsigned* matrix)
{
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    matrix[id] = id;
}

int main()
{
    dim3 grid(4, 1, 1);
    dim3 block(8, 1, 1);
    unsigned* matrix;
    unsigned* hmatrix;
    // 分配device記憶體
    cudaMalloc(&matrix, 4 * 8 * sizeof(unsigned));
    hmatrix = (unsigned*)malloc(4 * 8 * sizeof(unsigned));
    // 丟進device運算
    dkernel<<<grid, block>>>(matrix);
    // 把結果倒回host
    cudaMemcpy(hmatrix, matrix, 4 * 8 * sizeof(unsigned), cudaMemcpyDeviceToHost);

    for (unsigned i = 0; i < 4; i++) {
        for (unsigned j = 0; j < 8; j++) {
            printf("%2d", hmatrix[i * 8 + j]);
        }
        printf("\n");
    }
    return 0;
}
