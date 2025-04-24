ParallelImageProcessor:

Build four versions of an image processing program
Base (serial)
OpenMP (CPU multi-threaded)
MPI (distributed parallel)
Cuda (GPU parallel)
Compare runtime on multiple image sizes

Tools utilized:
C++
OpenCV -processes images and videos used to load, filter, and save images in the project.
CMake-Build tool that sets up and compiles the project — handles linking OpenCV, OpenMP, and MPI.
OpenMP
MPI
macOS M1
Nvidia Nsight Systems (for report)
Github

- apply common image enhancement techniques like: Gaussian blur and Canny 
edge detection
- Speed up processing using OpenMP for multi core parallelism
- Measure and compare performance so the execution time

