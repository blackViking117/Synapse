#include <iostream>
#include <preprocessor.h>
#include <filesystem>
#include <fstream>
#include <chrono>

//opencv include
#include <opencv2/opencv.hpp>
#include <vector>

#define CHECK_CUDA(call) \
do { \
    cudaError_t status = call; \
    if (status != cudaSuccess) { \
        std::cerr << "[CUDA ERROR] " << cudaGetErrorString(status) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

struct LetterboxInfo {
    float scale;
    int padX;
    int padY;
};


std::vector<unsigned char> convertMatToHWC(const cv::Mat& input, LetterboxInfo& info) {
    const int targetSize = 640;

    int originalWidth  = input.cols;
    int originalHeight = input.rows;

    // Preserve aspect ratio
    float scale = std::min(
        static_cast<float>(targetSize) / originalWidth,
        static_cast<float>(targetSize) / originalHeight
    );

    int newWidth  = static_cast<int>(std::round(originalWidth * scale));
    int newHeight = static_cast<int>(std::round(originalHeight * scale));

    // Resize
    cv::Mat resized;
    cv::resize(
        input,
        resized,
        cv::Size(newWidth, newHeight)
    );

    // Create padded image
    cv::Mat padded(
        targetSize,
        targetSize,
        CV_8UC3,
        cv::Scalar(114, 114, 114)
    );

    int padX = (targetSize - newWidth) / 2;
    int padY = (targetSize - newHeight) / 2;

    resized.copyTo(
        padded(cv::Rect(
            padX,
            padY,
            newWidth,
            newHeight
        ))
    );

    // BGR → RGB
    cv::cvtColor(
        padded,
        padded,
        cv::COLOR_BGR2RGB
    );

    info.scale = scale;
    info.padX = padX;
    info.padY = padY;


    // Ensure the matrix is continuous in memory before copying
    cv::Mat continuousMat;
    if (padded.isContinuous()) {
        continuousMat = padded;
    } else {
        continuousMat = padded.clone();
    }

    // Get the total number of bytes (Width * Height * Channels)
    size_t totalBytes = continuousMat.total() * continuousMat.elemSize();

    // Copy the flat data into a std::vector
    // Using unsigned char to prevent signed issues (-128 to 127) when passing to CUDA
    std::vector<unsigned char> hwcData(
        continuousMat.ptr<unsigned char>(), 
        continuousMat.ptr<unsigned char>() + totalBytes
    );

    return hwcData;
}


std::vector<float> imageProcessorPre(const cv::Mat& input, LetterboxInfo& info) {
    const int targetSize = 640;

    int originalWidth  = input.cols;
    int originalHeight = input.rows;

    // Preserve aspect ratio
    float scale = std::min(
        static_cast<float>(targetSize) / originalWidth,
        static_cast<float>(targetSize) / originalHeight
    );

    int newWidth  = static_cast<int>(std::round(originalWidth * scale));
    int newHeight = static_cast<int>(std::round(originalHeight * scale));

    // Resize
    cv::Mat resized;
    cv::resize(
        input,
        resized,
        cv::Size(newWidth, newHeight)
    );

    // Create padded image
    cv::Mat padded(
        targetSize,
        targetSize,
        CV_8UC3,
        cv::Scalar(114, 114, 114)
    );

    int padX = (targetSize - newWidth) / 2;
    int padY = (targetSize - newHeight) / 2;

    resized.copyTo(
        padded(cv::Rect(
            padX,
            padY,
            newWidth,
            newHeight
        ))
    );

    // BGR → RGB
    cv::cvtColor(
        padded,
        padded,
        cv::COLOR_BGR2RGB
    );

    // Store information needed for post-processing
    info.scale = scale;
    info.padX = padX;
    info.padY = padY;

    // HWC → CHW
    std::vector<float> tensor(3 * targetSize * targetSize);

    for (int y = 0; y < targetSize; ++y) {
        for (int x = 0; x < targetSize; ++x) {

            cv::Vec3b pixel = padded.at<cv::Vec3b>(y, x);

            // RGB
            tensor[
                0 * targetSize * targetSize +
                y * targetSize +
                x
            ] = pixel[0] / 255.0f;

            tensor[
                1 * targetSize * targetSize +
                y * targetSize +
                x
            ] = pixel[1] / 255.0f;

            tensor[
                2 * targetSize * targetSize +
                y * targetSize +
                x
            ] = pixel[2] / 255.0f;
        }
    }

    return tensor;
}

std::vector<std::string> imageList(std::string folder) {
	std::vector<std::string> locations; 
	auto allFiles = std::filesystem::recursive_directory_iterator(folder);

	for (auto item : allFiles) {
		if (std::filesystem::is_regular_file(item)) {
			locations.push_back(item.path().string());
		}
	}

	return locations;
}

int main() {
    std::string folderPath = "/media/swdep/500GB_SSD_PERCEP/black_magic/testData/testing/output_fp16/";
    std::cout << "[LOG] Scanning directory: " << folderPath << std::endl;
    std::vector<std::string> allImages = imageList(folderPath);
    std::cout << "[LOG] Found " << allImages.size() << " images." << std::endl;

    if (allImages.empty()) {
        std::cout << "[ERROR] No images found. Exiting." << std::endl;
        return 0;
    }

    // We will benchmark using the first image in the dataset to perform the 150 executions
    std::string item = allImages[0];
    std::cout << "[LOG] Using image for benchmark: " << item << std::endl;
    cv::Mat image = cv::imread(item);
    if (image.empty()) {
        std::cout << "[ERROR] Failed to read image." << std::endl;
        return 0;
    }

    void* input;
    void* output;
    int inSize = 640 * 640 * 3;
    int outSize = 640 * 640 * 3 * 4;

    CHECK_CUDA(cudaMalloc(&input, inSize));
    CHECK_CUDA(cudaMalloc(&output, outSize));

    LetterboxInfo info;
    std::vector<uint8_t> inputVal = convertMatToHWC(image, info);

    int warmup = 50;
    int runs = 1000;

    std::vector<double> times_v1(runs);
    std::vector<double> times_v2(runs);
    std::vector<double> times_v3(runs);
    std::vector<double> times_v4(runs);

    std::vector<float> out_v1(outSize / sizeof(float));
    std::vector<float> out_v2(outSize / sizeof(float));
    std::vector<float> out_v3(outSize / sizeof(float));
    std::vector<float> out_v4(outSize / sizeof(float));

    std::cout << "[LOG] Benchmarking Kernel V1 (50 Warmup, 100 Measured)..." << std::endl;
    for (int i = 0; i < warmup + runs; i++) {
        auto gpuStart = std::chrono::high_resolution_clock::now();
        std::vector<float> outputVal(outSize / sizeof(float));
        CHECK_CUDA(cudaMemcpy(input, reinterpret_cast<void*>(inputVal.data()), inSize, cudaMemcpyHostToDevice));
        imagePreprocessor<<<25, 1024>>>((uint8_t*)input, (float*)output);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(reinterpret_cast<void*>(outputVal.data()), output, outSize, cudaMemcpyDeviceToHost));
        auto gpuEnd = std::chrono::high_resolution_clock::now();
        
        if (i >= warmup) {
            times_v1[i - warmup] = std::chrono::duration<double, std::milli>(gpuEnd - gpuStart).count();
        }
        if (i == warmup + 70) {
            out_v1 = outputVal; // Save the 70th measured run
        }
    }

    std::cout << "[LOG] Benchmarking Kernel V2 (50 Warmup, 100 Measured)..." << std::endl;
    for (int i = 0; i < warmup + runs; i++) {
        auto gpuStart = std::chrono::high_resolution_clock::now();
        std::vector<float> outputVal(outSize / sizeof(float));
        CHECK_CUDA(cudaMemcpy(input, reinterpret_cast<void*>(inputVal.data()), inSize, cudaMemcpyHostToDevice));
        imagePreprocessor_v2<<<25, 1024>>>((uint8_t*)input, (float*)output);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(reinterpret_cast<void*>(outputVal.data()), output, outSize, cudaMemcpyDeviceToHost));
        auto gpuEnd = std::chrono::high_resolution_clock::now();
        
        if (i >= warmup) {
            times_v2[i - warmup] = std::chrono::duration<double, std::milli>(gpuEnd - gpuStart).count();
        }
        if (i == warmup + 70) {
            out_v2 = outputVal; // Save the 70th measured run
        }
    }

    std::cout << "[LOG] Benchmarking Kernel V3 (50 Warmup, 100 Measured)..." << std::endl;
    for (int i = 0; i < warmup + runs; i++) {
        auto gpuStart = std::chrono::high_resolution_clock::now();
        std::vector<float> outputVal(outSize / sizeof(float));
        CHECK_CUDA(cudaMemcpy(input, reinterpret_cast<void*>(inputVal.data()), inSize, cudaMemcpyHostToDevice));
        imagePreprocessor_v3<<<25, 1024>>>((uint8_t*)input, (float*)output);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(reinterpret_cast<void*>(outputVal.data()), output, outSize, cudaMemcpyDeviceToHost));
        auto gpuEnd = std::chrono::high_resolution_clock::now();
        
        if (i >= warmup) {
            times_v3[i - warmup] = std::chrono::duration<double, std::milli>(gpuEnd - gpuStart).count();
        }
        if (i == warmup + 70) {
            out_v3 = outputVal; // Save the 70th measured run
        }
    }

    std::cout << "[LOG] Benchmarking Kernel V4 (50 Warmup, 100 Measured)..." << std::endl;
    for (int i = 0; i < warmup + runs; i++) {
        auto gpuStart = std::chrono::high_resolution_clock::now();
        std::vector<float> outputVal(outSize / sizeof(float));
        CHECK_CUDA(cudaMemcpy(input, reinterpret_cast<void*>(inputVal.data()), inSize, cudaMemcpyHostToDevice));
        imagePreprocessor_v4<<<25, 1024>>>((uint8_t*)input, (float*)output);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(reinterpret_cast<void*>(outputVal.data()), output, outSize, cudaMemcpyDeviceToHost));
        auto gpuEnd = std::chrono::high_resolution_clock::now();
        
        if (i >= warmup) {
            times_v4[i - warmup] = std::chrono::duration<double, std::milli>(gpuEnd - gpuStart).count();
        }
        if (i == warmup + 70) {
            out_v4 = outputVal; // Save the 70th measured run
        }
    }

    std::cout << "[LOG] Running CPU Reference (1 Run, No Warmup)..." << std::endl;
    auto cpuStart = std::chrono::high_resolution_clock::now();
    std::vector<float> referenceVal = imageProcessorPre(image, info);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    double cpuTime = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();

    // Check Accuracy against the 70th runs
    std::cout << "[LOG] Validating GPU outputs against CPU reference..." << std::endl;
    bool pass_v1 = true, pass_v2 = true, pass_v3 = true, pass_v4 = true;
    for (size_t i = 0; i < referenceVal.size(); i++) {
        if (pass_v1 && std::abs(referenceVal[i] - out_v1[i]) > 1e-4) pass_v1 = false;
        if (pass_v2 && std::abs(referenceVal[i] - out_v2[i]) > 1e-4) pass_v2 = false;
        if (pass_v3 && std::abs(referenceVal[i] - out_v3[i]) > 1e-4) pass_v3 = false;
        if (pass_v4 && std::abs(referenceVal[i] - out_v4[i]) > 1e-4) pass_v4 = false;
        
        if (!pass_v1 && !pass_v2 && !pass_v3 && !pass_v4) break; // All failed, stop checking
    }

    // Calculate Medians
    std::sort(times_v1.begin(), times_v1.end());
    std::sort(times_v2.begin(), times_v2.end());
    std::sort(times_v3.begin(), times_v3.end());
    std::sort(times_v4.begin(), times_v4.end());

    double med_v1 = times_v1[runs / 2];
    double med_v2 = times_v2[runs / 2];
    double med_v3 = times_v3[runs / 2];
    double med_v4 = times_v4[runs / 2];

    double min_v1 = times_v1[0], max_v1 = times_v1[runs - 1];
    double min_v2 = times_v2[0], max_v2 = times_v2[runs - 1];
    double min_v3 = times_v3[0], max_v3 = times_v3[runs - 1];
    double min_v4 = times_v4[0], max_v4 = times_v4[runs - 1];

    double speedup_v1 = cpuTime / med_v1;
    double speedup_v2 = cpuTime / med_v2;
    double speedup_v3 = cpuTime / med_v3;
    double speedup_v4 = cpuTime / med_v4;

    double pct_v1 = ((cpuTime - med_v1) / cpuTime) * 100.0;
    double pct_v2 = ((cpuTime - med_v2) / cpuTime) * 100.0;
    double pct_v3 = ((cpuTime - med_v3) / cpuTime) * 100.0;
    double pct_v4 = ((cpuTime - med_v4) / cpuTime) * 100.0;

    std::cout << "\n======================================================================================================" << std::endl;
    std::cout << "                                        BENCHMARK RESULTS                                             " << std::endl;
    std::cout << "======================================================================================================" << std::endl;
    std::cout << "CPU Time (1 run): " << cpuTime << " ms" << std::endl;
    std::cout << "GPU V1 -> Min: " << min_v1 << " ms | Median: " << med_v1 << " ms | Max: " << max_v1 << " ms (Accuracy: " << (pass_v1 ? "PASS" : "FAIL") << ") | Speedup: " << speedup_v1 << "x (" << pct_v1 << "% faster)" << std::endl;
    std::cout << "GPU V2 -> Min: " << min_v2 << " ms | Median: " << med_v2 << " ms | Max: " << max_v2 << " ms (Accuracy: " << (pass_v2 ? "PASS" : "FAIL") << ") | Speedup: " << speedup_v2 << "x (" << pct_v2 << "% faster)" << std::endl;
    std::cout << "GPU V3 -> Min: " << min_v3 << " ms | Median: " << med_v3 << " ms | Max: " << max_v3 << " ms (Accuracy: " << (pass_v3 ? "PASS" : "FAIL") << ") | Speedup: " << speedup_v3 << "x (" << pct_v3 << "% faster)" << std::endl;
    std::cout << "GPU V4 -> Min: " << min_v4 << " ms | Median: " << med_v4 << " ms | Max: " << max_v4 << " ms (Accuracy: " << (pass_v4 ? "PASS" : "FAIL") << ") | Speedup: " << speedup_v4 << "x (" << pct_v4 << "% faster)" << std::endl;
    std::cout << "======================================================================================================" << std::endl;

    cudaFree(input);
    cudaFree(output);
}