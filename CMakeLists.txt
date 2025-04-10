cmake_minimum_required(VERSION 3.15)
project(ImageEnhancementProject)

set(CMAKE_CXX_STANDARD 17)

# --- OpenCV Setup ---
set(OpenCV_DIR /opt/homebrew/opt/opencv/share/opencv4)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(/opt/homebrew/lib)

# --- Base Serial Version ---
add_executable(base base/main.cpp)
target_link_libraries(base ${OpenCV_LIBS})

# --- OpenMP Version ---
add_executable(openmp openmp/main.cpp)
target_include_directories(openmp PRIVATE /opt/homebrew/include)
target_compile_options(openmp PRIVATE -Xpreprocessor -fopenmp)
target_link_libraries(openmp ${OpenCV_LIBS} /opt/homebrew/lib/libomp.dylib)

# --- MPI Version ---
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
add_executable(mpi mpi/main.cpp)
target_link_libraries(mpi ${OpenCV_LIBS} MPI::MPI_CXX)
# --- CUDA Version (Optional / Demo) ---
# Only enable this on machines with NVIDIA GPU
# enable_language(CUDA)
# find_package(CUDA REQUIRED)
# add_executable(cuda cuda/main.cu)
# target_link_libraries(cuda ${OpenCV_LIBS} cuda)

