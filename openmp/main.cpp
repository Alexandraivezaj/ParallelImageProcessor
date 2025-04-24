#include <opencv2/opencv.hpp>
#include <iostream>
#include <omp.h>    
#include <chrono>   

int main() {
    cv::Mat input = cv::imread("../data/sample.jpg", cv::IMREAD_COLOR);
    if (input.empty()) {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }

    auto start = std::chrono::high_resolution_clock::now();  // start

    cv::Mat blurred;
    cv::GaussianBlur(input, blurred, cv::Size(5, 5), 1.5);  // opencv's optimized blur

    cv::Mat edges;
    cv::Canny(blurred, edges, 100, 200);                    // edge detection

    auto end = std::chrono::high_resolution_clock::now();  // end

    cv::imwrite("../results/openmp_edges.jpg", edges);
    std::cout << "Saved to results/openmp_edges.jpg" << std::endl;

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "OpenMP (optimized OpenCV) processing time: " << duration.count() << " ms" << std::endl;

    return 0;
}

