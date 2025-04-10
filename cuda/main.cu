#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>

// Dummy kernel
__global__ void dummyKernel(unsigned char* data) {
    // No-op
}

int main() {
    cv::Mat input = cv::imread("../data/sample.jpg", cv::IMREAD_COLOR);
    if (input.empty()) {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }

    cv::Mat gray;
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);

    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.5);

    unsigned char* d_data;
    size_t dataSize = blurred.total() * blurred.elemSize();
    cudaMalloc((void**)&d_data, dataSize);
    cudaMemcpy(d_data, blurred.data, dataSize, cudaMemcpyHostToDevice);
    dummyKernel<<<1, 1>>>(d_data);
    cudaMemcpy(blurred.data, d_data, dataSize, cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    cv::Mat edges;
    cv::Canny(blurred, edges, 100, 200);
    cv::imwrite("../results/cuda_edges.jpg", edges);
    std::cout << "✅ CUDA (stub) output saved to results/cuda_edges.jpg" 
<< std::endl;

    return 0;
}

