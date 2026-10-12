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
};

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


    void* input;
    void* output;
    size_t inputSize = batchSize * imageElements * sizeof(float);
    size_t outputSize = batchSize * outputPerImage * sizeof(float);

    cudaHostAlloc(&input,  inputSize,  cudaHostAllocMapped);
    cudaHostAlloc(&output, outputSize, cudaHostAllocMapped);

    cudaStream_t preProcessorStream;
    cudaStreamCreate(&preProcessorStream);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    context->setTensorAddress(inputName, input);
    context->setTensorAddress(outputName, output);
    
    uint8_t* d_hwc;
    cudaMalloc(&d_hwc, 640 * 640 * 3 * sizeof(uint8_t));

    int counter = 1;
    std::cout << std::endl;
    
    int numOfImgs = allImages.size();

    for (int i = 0; i+3 < numOfImgs; i += 4) {
        if (counter % 4 == 0) {
            std::cout << "Iteration: " << counter << std::endl;
        }
        
        auto item1 = allImages[i];
        auto item2 = allImages[i + 1];
        auto item3 = allImages[i + 2];
        auto item4 = allImages[i + 3];
        LetterboxInfo info;
		cv::Mat image1 = cv::imread(item1);
        cv::Mat image2 = cv::imread(item2);
        cv::Mat image3 = cv::imread(item3);
        cv::Mat image4 = cv::imread(item4);
		
		std::vector<unsigned char> inputHWC1 = convertMatToHWC(image1, info);
        std::vector<unsigned char> inputHWC2 = convertMatToHWC(image2, info);
        std::vector<unsigned char> inputHWC3 = convertMatToHWC(image3, info);
        std::vector<unsigned char> inputHWC4 = convertMatToHWC(image4, info);

		cudaMemcpyAsync(d_hwc_1, inputHWC1.data(), inputHWC1.size(), cudaMemcpyHostToDevice, preProcessorStream1);
		imagePreprocessor_v4<<<25, 1024, 0, preProcessorStream1>>>(d_hwc_1, (float*)input);

        cudaMemcpyAsync(d_hwc_2, inputHWC2.data(), inputHWC2.size(), cudaMemcpyHostToDevice, preProcessorStream2);
		imagePreprocessor_v4<<<25, 1024, 0, preProcessorStream2>>>(d_hwc_2, ((float*)input + (3 * 640 * 640)));

        cudaMemcpyAsync(d_hwc_3, inputHWC3.data(), inputHWC3.size(), cudaMemcpyHostToDevice, preProcessorStream3);
		imagePreprocessor_v4<<<25, 1024, 0, preProcessorStream3>>>(d_hwc_3, ((float*)input + (3 * 640 * 640 * 2)));

        cudaMemcpyAsync(d_hwc_4, inputHWC4.data(), inputHWC4.size(), cudaMemcpyHostToDevice, preProcessorStream4);
		imagePreprocessor_v4<<<25, 1024, 0, preProcessorStream4>>>(d_hwc_4, ((float*)input + (3 * 640 * 640 * 3)));

        cudaStreamSynchronize(preProcessorStream1);
        cudaStreamSynchronize(preProcessorStream2);
        cudaStreamSynchronize(preProcessorStream3);
        cudaStreamSynchronize(preProcessorStream4);
		
		if (!context->enqueueV3(stream)) {
			std::cout << "enqueueV3 Failed.....\n";
			return -1;
		}
	
		cudaStreamSynchronize(stream);
        counter++;
	}

    std::cout << "Last iteration: " << counter << std::endl;

    cudaFree(d_hwc_1);
    cudaFree(d_hwc_2);
    cudaFree(d_hwc_3);
    cudaFree(d_hwc_4);
    cudaStreamDestroy(preProcessorStream1);
    cudaStreamDestroy(preProcessorStream2);
    cudaStreamDestroy(preProcessorStream3);
    cudaStreamDestroy(preProcessorStream4);
    cudaStreamDestroy(stream);
    cudaFreeHost(input);
    cudaFreeHost(output);


    return 0;
}