#include <opencv2/opencv.hpp>
#include <iostream>
#include <omp.h>    // still included in case we want to parallelize later
#include <chrono>   // for timing

int main() {
    cv::Mat input = cv::imread("../data/sample.jpg", cv::IMREAD_COLOR);
    if (input.empty()) {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }

    auto start = std::chrono::high_resolution_clock::now();  // ⏱️ Start

    cv::Mat blurred;
    cv::GaussianBlur(input, blurred, cv::Size(5, 5), 1.5);  // OpenCV's optimized blur

    cv::Mat edges;
    cv::Canny(blurred, edges, 100, 200);                    // Edge detection

    auto end = std::chrono::high_resolution_clock::now();  // ⏱️ End

    cv::imwrite("../results/openmp_edges.jpg", edges);
    std::cout << "Saved to results/openmp_edges.jpg" << std::endl;

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "OpenMP (optimized OpenCV) processing time: " << duration.count() << " ms" << std::endl;

    return 0;
}

