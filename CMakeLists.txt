cmake_minimum_required(VERSION 3.15)
project(ImageEnhancementProject)

set(CMAKE_CXX_STANDARD 17)

# OpenCV Setup
set(OpenCV_DIR /opt/homebrew/opt/opencv/share/opencv4)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(/opt/homebrew/lib)

# --- Base Serial Version ---
add_executable(base base/main.cpp)
target_link_libraries(base ${OpenCV_LIBS})

# --- OpenMP Version ---
add_executable(openmp openmp/main.cpp)

# Apply OpenMP flags ONLY to the OpenMP target
target_include_directories(openmp PRIVATE /opt/homebrew/include)
target_compile_options(openmp PRIVATE -Xpreprocessor -fopenmp)
target_link_libraries(openmp ${OpenCV_LIBS} /opt/homebrew/lib/libomp.dylib)
