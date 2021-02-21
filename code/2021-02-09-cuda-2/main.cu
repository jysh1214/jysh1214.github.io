#include <stdio.h>
#include <cuda.h>

__global__ void square(unsigned* matrix, unsigned* result, unsigned matrixsize)
{
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned jj = 0; jj < matrixsize; ++jj) {
        for (unsigned kk = 0; kk < matrixsize; ++kk) {
            result[id * matrixsize + jj] +=
                matrix[id * matrixsize + kk] * matrix[kk * matrixsize + jj];
        }
    }
}

#define N 1000000
#define M 1000

int main(void)
{
    unsigned* hmatrix = (unsigned*)malloc(N * sizeof(unsigned));
    for (unsigned i = 0; i < N; ++i) {
        hmatrix[i] = i % 10;
    }

    unsigned* hresult = (unsigned*)malloc(N * sizeof(unsigned));

    unsigned* dmatrix;
    unsigned* dresult;

    cudaMalloc(&dmatrix, N * sizeof(unsigned));
    cudaMalloc(&dresult, N * sizeof(unsigned));

    cudaMemcpy(dmatrix, hmatrix, N * sizeof(unsigned), cudaMemcpyHostToDevice);

    dim3 grid(M, 1, 1);
    dim3 block(M, 1, 1);

    square<<<grid, block>>>(dmatrix, dresult, M);
    cudaMemcpy(hresult, dresult, N * sizeof(unsigned), cudaMemcpyDeviceToHost);

    free(hmatrix);
    free(hresult);

    cudaFree(dmatrix);
    cudaFree(dresult);

    return 0;
}
