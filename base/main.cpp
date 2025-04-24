#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>  

int main() {
    cv::Mat input = cv::imread("../data/sample.jpg", cv::IMREAD_COLOR);
    if (input.empty()) {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }

    auto start = std::chrono::high_resolution_clock::now();  // starts

    cv::Mat blurred;
    cv::GaussianBlur(input, blurred, cv::Size(5, 5), 1.5);

    cv::Mat edges;
    cv::Canny(blurred, edges, 100, 200);

    auto end = std::chrono::high_resolution_clock::now();  // ends

    cv::imwrite("../results/base_edges.jpg", edges);
    std::cout << "Saved to results/base_edges.jpg" << std::endl;

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Base (serial) processing time: " << duration.count() << " ms" << std::endl;

    return 0;
}
