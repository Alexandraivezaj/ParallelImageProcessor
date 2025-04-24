#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>  

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cv::Mat inputImage;

    int rows, cols, type;
    int sliceRows;

    if (rank == 0) {
        inputImage = cv::imread("../data/sample.jpg", cv::IMREAD_COLOR);
        if (inputImage.empty()) {
            cerr << "Error loading image!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        rows = inputImage.rows;
        cols = inputImage.cols;
        type = inputImage.type();

        if (rows % size != 0) {
            cerr << "Image height must be divisible by number of processes!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        sliceRows = rows / size;
    }

    // broadcast metadata to all processes
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&type, 1, MPI_INT, 0, MPI_COMM_WORLD);
    sliceRows = rows / size;

    //  want to start timing after sync
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    int sliceSize = sliceRows * cols * 3;
    std::vector<uchar> localData(sliceSize);

    if (rank == 0) {
        std::vector<uchar> imageData(rows * cols * 3);
        std::memcpy(imageData.data(), inputImage.data, imageData.size());
        MPI_Scatter(imageData.data(), sliceSize, MPI_UNSIGNED_CHAR,
                    localData.data(), sliceSize, MPI_UNSIGNED_CHAR,
                    0, MPI_COMM_WORLD);
    } else {
        MPI_Scatter(nullptr, sliceSize, MPI_UNSIGNED_CHAR,
                    localData.data(), sliceSize, MPI_UNSIGNED_CHAR,
                    0, MPI_COMM_WORLD);
    }
// want to process loacal slice
    cv::Mat localSlice(sliceRows, cols, CV_8UC3, localData.data());

    cv::Mat blurred, edges;
    cv::GaussianBlur(localSlice, blurred, cv::Size(5, 5), 1.5);
    cv::Canny(blurred, edges, 100, 200);

    std::vector<uchar> processedSlice;
    if (edges.isContinuous()) {
        processedSlice.assign(edges.data, edges.data + edges.total());
    } else {
        cerr << "Non-continuous memory — unexpected!" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::vector<uchar> fullImage;
    if (rank == 0) {
        fullImage.resize(rows * cols);
    }

    MPI_Gather(processedSlice.data(), sliceRows * cols, MPI_UNSIGNED_CHAR,
               fullImage.data(), sliceRows * cols, MPI_UNSIGNED_CHAR,
               0, MPI_COMM_WORLD);

    // end timing 
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    if (rank == 0) {
        cv::Mat outputImage(rows, cols, CV_8UC1, fullImage.data());
        cv::imwrite("../results/mpi_edges.jpg", outputImage);
        cout << "✅ MPI output saved to results/mpi_edges.jpg" << endl;

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        cout << "🕒 MPI processing time: " << duration.count() << " ms" << endl;
    }

    MPI_Finalize();
    return 0;
}
