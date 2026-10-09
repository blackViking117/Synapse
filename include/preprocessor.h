#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#include <cuda_runtime.h>
#include <cstdint>

__global__ void imagePreprocessor(uint8_t* imageHWC, float* imageCHW);
__global__ void imagePreprocessor_v2(uint8_t* imageHWC, float* imageCHW);
__global__ void imagePreprocessor_v3(uint8_t* imageHWC, float* imageCHW);
__global__ void imagePreprocessor_v4(uint8_t* imageHWC, float* imageCHW);

#endif // PREPROCESSOR_H
