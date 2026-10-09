#include <iostream>
#include <cstdint>

#define COARSE_FACTOR 16
#define IMG_DIM 640
#define TILE_WIDTH 16384

__global__ void imagePreprocessor(uint8_t* imageHWC, float* imageCHW) {

    __shared__ uint8_t HWC[TILE_WIDTH][3];

    int bx = blockIdx.x;
    int tx = threadIdx.x;

    int localPix = tx * COARSE_FACTOR;
    int globalPix = bx * (TILE_WIDTH) + localPix;

    //collabratively loading data into shared memory
    for (int i = 0; i < COARSE_FACTOR; i++) {
        HWC[tx + (blockDim.x * i)][0] = imageHWC[((bx * (TILE_WIDTH)) + (tx + (blockDim.x * i)))*3];
        HWC[tx + (blockDim.x * i)][1] = imageHWC[(((bx * (TILE_WIDTH)) + (tx + (blockDim.x * i)))*3) + 1];
        HWC[tx + (blockDim.x * i)][2] = imageHWC[(((bx * (TILE_WIDTH)) + (tx + (blockDim.x * i)))*3) + 2];
    }
    __syncthreads();


    //performing actual calulation and writing to global memory
    for (int i = 0; i < COARSE_FACTOR; i++) {
        imageCHW[((bx * (TILE_WIDTH)) + (tx + (blockDim.x * i)))] = HWC[tx + (blockDim.x * i)][0] / 255.0f;
        imageCHW[((bx * (TILE_WIDTH)) + (tx + (blockDim.x * i))) + (IMG_DIM * IMG_DIM)] = HWC[tx + (blockDim.x * i)][1] / 255.0f;
        imageCHW[((bx * (TILE_WIDTH)) + (tx + (blockDim.x * i))) + (IMG_DIM * IMG_DIM * 2)] = HWC[tx + (blockDim.x * i)][2] / 255.0f;
    }
}

__global__ void imagePreprocessor_v2(uint8_t* imageHWC, float* imageCHW) {

    __shared__ uint8_t HWC[TILE_WIDTH][3];

    int bx = blockIdx.x;
    int tx = threadIdx.x;

    int globalPixPart = bx * (TILE_WIDTH);

    //collabratively loading data into shared memory
    for (int i = 0; i < COARSE_FACTOR; i++) {
        int localPix = tx + (blockDim.x * i);
        int globalPix = globalPixPart + localPix;

        HWC[localPix][0] = imageHWC[globalPix*3];
        HWC[localPix][1] = imageHWC[(globalPix*3) + 1];
        HWC[localPix][2] = imageHWC[(globalPix*3) + 2];
    }
    __syncthreads();


    //performing actual calulation and writing to global memory
    for (int i = 0; i < COARSE_FACTOR; i++) {
        int localPix = tx + (blockDim.x * i);
        int globalPix = globalPixPart + localPix;


        imageCHW[globalPix] = HWC[tx + (blockDim.x * i)][0] / 255.0f;
        imageCHW[globalPix + (IMG_DIM * IMG_DIM)] = HWC[tx + (blockDim.x * i)][1] / 255.0f;
        imageCHW[globalPix + (IMG_DIM * IMG_DIM * 2)] = HWC[tx + (blockDim.x * i)][2] / 255.0f;
    }
}

__global__ void imagePreprocessor_v3(uint8_t* imageHWC, float* imageCHW) {

    __shared__ uint8_t HWC[TILE_WIDTH][3];

    int bx = blockIdx.x;
    int tx = threadIdx.x;

    int globalBytePart = bx * (TILE_WIDTH) * 3;

    uint8_t* HWC_flat = (uint8_t*)HWC;

    //collabratively loading data into shared memory
    for (int i = 0; i < 48; i++) {
        int localByte = tx + (blockDim.x * i);
        int globalByte = globalBytePart + localByte;

        HWC_flat[localByte] = imageHWC[globalByte];
    }
    __syncthreads();


    //performing actual calulation and writing to global memory
    int globalPixPart = bx * (TILE_WIDTH);
    for (int i = 0; i < COARSE_FACTOR; i++) {
        int localPix = tx + (blockDim.x * i);
        int globalPix = globalPixPart + localPix;

        imageCHW[globalPix] = HWC[localPix][0] / 255.0f;
        imageCHW[globalPix + (IMG_DIM * IMG_DIM)] = HWC[localPix][1] / 255.0f;
        imageCHW[globalPix + (IMG_DIM * IMG_DIM * 2)] = HWC[localPix][2] / 255.0f;
    }
}

__global__ void imagePreprocessor_v4(uint8_t* imageHWC, float* imageCHW) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;

    int globalPixPart = bx * (TILE_WIDTH);


    //performing actual calulation and writing to global memory
    for (int i = 0; i < COARSE_FACTOR; i++) {
        int localPix = tx + (blockDim.x * i);
        int globalPix = globalPixPart + localPix;


        imageCHW[globalPix] = imageHWC[globalPix*3] / 255.0f;
        imageCHW[globalPix + (IMG_DIM * IMG_DIM)] = imageHWC[(globalPix*3) + 1] / 255.0f;
        imageCHW[globalPix + (IMG_DIM * IMG_DIM * 2)] = imageHWC[(globalPix*3) + 2] / 255.0f;
    }
}