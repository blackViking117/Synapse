#include <iostream>
#include <preprocessor.h>

//opencv include
#include <opencv2/opencv.hpp>
#include <vector>
#include <thread>

struct LetterboxInfo {
    float scale;
    int padX;
    int padY;
    int newWidth;
    int newHeight;
};

LetterboxInfo computeLetterboxInfo(int originalWidth, int originalHeight, int targetSize = 640) {
    LetterboxInfo info;
    info.scale = std::min(
        static_cast<float>(targetSize) / originalWidth,
        static_cast<float>(targetSize) / originalHeight
    );
    info.newWidth  = static_cast<int>(std::round(originalWidth * info.scale));
    info.newHeight = static_cast<int>(std::round(originalHeight * info.scale));
    info.padX = (targetSize - info.newWidth) / 2;
    info.padY = (targetSize - info.newHeight) / 2;
    return info;
}

#include <fstream>
#include <filesystem>
#include <NvInfer.h>

using namespace nvinfer1;

std::vector<std::string> imageList(const std::string& folder) {
	std::vector<std::string> locations; 
	auto allFiles = std::filesystem::recursive_directory_iterator(folder);
	for (auto item : allFiles) {
		if (std::filesystem::is_regular_file(item)) {
			locations.push_back(item.path().string());
		}
	}
	return locations;
}

std::string imagePath2Name(const std::string& path) {
    std::filesystem::path p(path);
    return p.stem().string();
}

cv::Mat PostProcessor(float* output, size_t outputCount, const cv::Mat& originalImage, const LetterboxInfo& info) {
    // Dummy implementation for now, just returning the original image
    return originalImage.clone();
}

namespace {

// Standard 80 COCO class names, in the order YOLOv8 was trained on.
const std::vector<std::string> kCocoClassNames = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush"
};

}


//necessary logger for CUDA
class Logger : public ILogger
{
    void log(Severity severity, char const* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
            std::cout << "[TRT]: " << msg << std::endl;
        }
    }
} logger;


std::vector<unsigned char> convertMatToHWC(const cv::Mat& input, const LetterboxInfo& info) {
    const int targetSize = 640;

    // Resize using precomputed dimensions
    cv::Mat resized;
    cv::resize(
        input,
        resized,
        cv::Size(info.newWidth, info.newHeight)
    );

    // Create padded image
    cv::Mat padded(
        targetSize,
        targetSize,
        CV_8UC3,
        cv::Scalar(114, 114, 114)
    );

    resized.copyTo(
        padded(cv::Rect(
            info.padX,
            info.padY,
            info.newWidth,
            info.newHeight
        ))
    );

    // BGR → RGB
    cv::cvtColor(
        padded,
        padded,
        cv::COLOR_BGR2RGB
    );


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

int main() {
    std::string folderPath = "/media/swdep/500GB_SSD_PERCEP/black_magic/testData/testing/output_fp16/";
    std::string outputPath = "/media/swdep/500GB_SSD_PERCEP/black_magic/testData/testing/output_main/";
    
    // Ensure output directory exists
    std::filesystem::create_directories(outputPath);

    std::string engineFilePath = "/media/swdep/500GB_SSD_PERCEP/black_magic/yolov8x_b4.engine";

    std::ifstream engineFile(engineFilePath, std::ios::binary | std::ios::ate);
	if (!engineFile) {
		std::cout << "Couldn't Open the Engine File." << std::endl;
		return -1;
	}


	const int batchSize = 4;
    const int imageElements = 3 * 640 * 640;


	size_t engineSize = static_cast<size_t>(engineFile.tellg());
	engineFile.seekg(0, std::ios::beg);

	std::vector<char> engineData(engineSize);
	engineFile.read(engineData.data(), engineSize);


	IRuntime* runtime = createInferRuntime(logger);

	ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineSize);

	if (!engine) {
		std::cout << "Engine deserialization Failue." << std::endl;
		return -1;
	}

	IExecutionContext *context = engine->createExecutionContext();

	//gathering all the image locations
	std::vector<std::string> allImages = imageList(folderPath);

    const char* inputName = engine->getIOTensorName(0);
    const char* outputName = engine->getIOTensorName(1);
    context->setInputShape(inputName, Dims4{batchSize, 3, 640, 640});

    Dims const outputDims = context->getTensorShape(outputName);
    // For YOLOv8, outputDims is [batchSize, 84, 8400]
    size_t outputPerImage = 1;
    for (int32_t i = 1; i < outputDims.nbDims; ++i) { // start from dim 1 (skip batch)
        outputPerImage *= outputDims.d[i];
    }


    void* input[2];
    void* output[2];
    size_t inputSize = batchSize * imageElements * sizeof(float);
    size_t outputSize = batchSize * outputPerImage * sizeof(float);

    cudaHostAlloc(&input[0],  inputSize,  cudaHostAllocMapped);
    cudaHostAlloc(&output[0], outputSize, cudaHostAllocMapped);
    cudaHostAlloc(&input[1],  inputSize,  cudaHostAllocMapped);
    cudaHostAlloc(&output[1], outputSize, cudaHostAllocMapped);

    cudaStream_t prepStream[2], inferStream[2];
    for (int s = 0; s < 2; ++s) {
        cudaStreamCreate(&prepStream[s]);
        cudaStreamCreate(&inferStream[s]);
    }

    context->setTensorAddress(inputName, input[0]);
    context->setTensorAddress(outputName, output[0]);
    
    uint8_t* d_hwc[2];
    cudaMalloc(&d_hwc[0], 640 * 640 * 3 * sizeof(uint8_t));
    cudaMalloc(&d_hwc[1], 640 * 640 * 3 * sizeof(uint8_t));
    
    int numOfImgs = allImages.size();
    if (numOfImgs == 0) {
        std::cerr << "No images found!\n";
        return -1;
    }

    // 4 concurrent CPU threads for imread + convertMatToHWC (read-only access to staticLetterboxInfo)
    std::thread workers[4];

    // Precalculate LetterboxInfo ONCE from the first image (read-only for all threads)
    cv::Mat firstImg = cv::imread(allImages[0]);
    if (firstImg.empty()) {
        std::cerr << "Failed to read first image for letterbox info!\n";
        return -1;
    }
    const LetterboxInfo staticLetterboxInfo = computeLetterboxInfo(firstImg.cols, firstImg.rows, 640);

    int slot = 0;
    
    std::string items[4] = {allImages[0], allImages[1], allImages[2], allImages[3]};
    std::vector<unsigned char> inputHWCs[4];

    for (int b = 0; b < 4; ++b) {
        workers[b] = std::thread([b, &items, &inputHWCs, &staticLetterboxInfo]() {
            cv::Mat img = cv::imread(items[b]);
            inputHWCs[b] = convertMatToHWC(img, staticLetterboxInfo);
        });
    }
    for (int b = 0; b < 4; ++b) {
        workers[b].join();
    }

    // Single preprocessing stream sequentially queues H2D copies and kernels
    for (int b = 0; b < 4; ++b) {
        cudaMemcpyAsync(d_hwc[0], inputHWCs[b].data(), inputHWCs[b].size(), cudaMemcpyHostToDevice, prepStream[0]);
        imagePreprocessor_v4<<<25, 1024, 0, prepStream[0]>>>(d_hwc[0], (float*)input[0] + (b * imageElements));
    }

    cudaStreamSynchronize(prepStream[0]);
    
    if (!context->enqueueV3(inferStream[0])) {
        std::cout << "enqueueV3 Failed.....\n";
        return -1;
    }

    for (int i = 4; i+3 < numOfImgs; i += 4) {

        int curr = slot;
        int next = 1 - slot;

        std::string nextItems[4] = {allImages[i], allImages[i + 1], allImages[i + 2], allImages[i + 3]};

        for (int b = 0; b < 4; ++b) {
            workers[b] = std::thread([b, &nextItems, &inputHWCs, &staticLetterboxInfo]() {
                cv::Mat img = cv::imread(nextItems[b]);
                inputHWCs[b] = convertMatToHWC(img, staticLetterboxInfo);
            });
        }


        for (int b = 0; b < 4; ++b) {
            workers[b].join();
        }

        // Preprocess next batch into `next` slot (while GPU runs `curr` slot)
        for (int b = 0; b < 4; ++b) {
            cudaMemcpyAsync(d_hwc[next], inputHWCs[b].data(), inputHWCs[b].size(), cudaMemcpyHostToDevice, prepStream[next]);
            imagePreprocessor_v4<<<25, 1024, 0, prepStream[next]>>>(d_hwc[next], (float*)input[next] + (b * imageElements));
        }

        // Wait for: (1) preprocessing of next batch done, (2) inference of curr batch done
        // BOTH must finish before we re-point context addresses and launch next inference
        cudaStreamSynchronize(prepStream[next]);
        cudaStreamSynchronize(inferStream[curr]);

        context->setTensorAddress(inputName,  input[next]);
        context->setTensorAddress(outputName, output[next]);

        context->enqueueV3(inferStream[next]);  // GPU starts next batch
        slot = next;

	}

    cudaStreamSynchronize(inferStream[slot]);


    cudaFree(d_hwc[0]);
    cudaFree(d_hwc[1]);
    cudaStreamDestroy(prepStream[0]);
    cudaStreamDestroy(prepStream[1]);
    cudaStreamDestroy(inferStream[0]);
    cudaStreamDestroy(inferStream[1]);
    cudaFreeHost(input[0]);
    cudaFreeHost(input[1]);
    cudaFreeHost(output[0]);
    cudaFreeHost(output[1]);


    return 0;
}