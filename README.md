# ParallelImageProcessor

A C++ image processing project that compares the performance of **serial 
(base)** vs **parallel (OpenMP)** implementations using OpenCV.

## 🚀 Project Goals

- Apply common image enhancement techniques (e.g., Gaussian blur and Canny 
edge detection)
- Speed up processing using OpenMP for multi-core parallelism
- Measure and compare performance (execution time)

## 🛠️ How to Build

Make sure you have OpenCV and CMake installed (with Homebrew on macOS):

```bash
brew install opencv libomp cmake

