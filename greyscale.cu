#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <npp.h>
#include <nppi_color_conversion.h>
#include <thread>
#include <atomic>
#include <iostream>

// --- Global Config ---
std::atomic<bool> running(true);

// Double buffer CPU pageable
cv::Mat bufferMat[2];
std::atomic<int> matIndex(0);

// Double buffer pinned
cv::Mat *pinnedBuffer[2];
std::atomic<int> pinnedMatIndex(0);
int pinnedWriteIndex = 0;

int width = 0;
int height = 0;
int type = CV_8UC3;

// --- Pinned Mem Alloc Helper Func ---
cv::Mat *allocate_pinned(size_t bytesize)
{
    void *ptr;
    cudaError_t err = cudaHostAlloc(&ptr, bytesize, cudaHostAllocDefault);
    if (err != cudaSuccess)
    {
        std::cerr << "Errore allocazione pinned memory!" << std::endl;
        return nullptr;
    }
    return new cv::Mat(height, width, type, ptr);
}

// --- Capture thread ---
void capture_thread(cv::VideoCapture &cap)
{
    cv::Mat frame;
    int writeIndex = 0;

    while (running.load())
    {
        if (!cap.read(frame))
            continue;

        {
            frame.copyTo(bufferMat[writeIndex]);
        }

        writeIndex = 1 - writeIndex;
        matIndex.store(writeIndex);
    }
}

// --- Pinned copy thread ---
void pinned_copy_thread()
{
    while (running.load())
    {
        int readIndex = 1 - matIndex.load(); 
        cv::Mat *dst = pinnedBuffer[pinnedWriteIndex];

        {
            bufferMat[readIndex].copyTo(*dst);
        }

        // Swap pinned buffer
        pinnedWriteIndex = 1 - pinnedWriteIndex;
        pinnedMatIndex.store(pinnedWriteIndex);
    }
}

// --- Async Copy host->device ---
// As of now ASYNC brings NO ADVANTAGES because we are on a single stream which is sequential by def
void copyPinnedToDevice(uint8_t *d_frame, cudaStream_t stream, size_t bytesize)
{
    int gpuReadIndex = 1 - pinnedMatIndex.load(std::memory_order_acquire);
    cv::Mat *src = pinnedBuffer[gpuReadIndex];

    cudaMemcpyAsync(d_frame, src->data, bytesize, cudaMemcpyHostToDevice, stream);
}

// --- Buffer GPU ---
std::tuple<Npp8u *, float *> allocatePreprocessBuffers()
{
    Npp8u *d_gray = nullptr;
    float *d_grayf = nullptr;

    cudaError_t err = cudaMalloc(&d_gray, width * height * sizeof(Npp8u));
    if (err != cudaSuccess)
    {
        std::cerr << "Errore allocazione d_gray: " << cudaGetErrorString(err) << std::endl;
        return {nullptr, nullptr};
    }

    err = cudaMalloc(&d_grayf, width * height * sizeof(float));
    if (err != cudaSuccess)
    {
        std::cerr << "Errore allocazione d_grayf: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_gray);
        return {nullptr, nullptr};
    }

    return {d_gray, d_grayf};
}

// --- Grayscale on GPU ---
void greyScale(uint8_t *d_frame, Npp8u *d_gray)
{
    NppiSize roi{width, height};
    const Npp32f coeffs[3] = {0.114f, 0.587f, 0.299f};

    nppiColorToGray_8u_C3C1R(d_frame, width * 3, d_gray, width, roi, coeffs);
}

// --- Copy device->host ---
void deviceToHost(Npp8u *d_gray, cv::Mat &hostGray)
{
    size_t bytesize = width * height * sizeof(Npp8u);
    cudaMemcpy(hostGray.data, d_gray, bytesize, cudaMemcpyDeviceToHost);
}

// --- MAIN ---
int main(int argc, char* argv[])
{
    cv::VideoCapture cap(0, cv::CAP_V4L2);
    if (!cap.isOpened())
    {
        std::cerr << "Impossible to open webcam" << std::endl;
        return -1;
    }
    if (argc > 2){
        try
        {
            int width = std::stoi(argv[1]);
            int height = std::stoi(argv[2]);
            cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
            cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            return 1;
        }
    }

    width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    size_t bytesize = width * height * CV_ELEM_SIZE(type);
    std::cout << "Webcam: " << width << "x" << height << std::endl;

    bufferMat[0] = cv::Mat(height, width, type);
    bufferMat[1] = cv::Mat(height, width, type);

    pinnedBuffer[0] = allocate_pinned(bytesize);
    pinnedBuffer[1] = allocate_pinned(bytesize);
    if (!pinnedBuffer[0] || !pinnedBuffer[1])
        return -1;

    std::thread tCapture(capture_thread, std::ref(cap));
    std::thread tPinned(pinned_copy_thread);

    Npp8u *d_gray = nullptr;
    float *d_grayf = nullptr;
    std::tie(d_gray, d_grayf) = allocatePreprocessBuffers();
    if (!d_gray || !d_grayf)
        return -1;

    uint8_t *d_frame = nullptr;
    cudaMalloc(&d_frame, bytesize);

    cv::Mat grayMat(height, width, CV_8UC1);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    nppSetStream(stream);

    int frameCount = 0;

    while (true)
    {
        auto tStart = std::chrono::high_resolution_clock::now();

        // H→D transfer
        copyPinnedToDevice(d_frame, stream, bytesize);
        //cudaStreamSynchronize(stream);

        // Grayscale kernel
        greyScale(d_frame, d_gray);
        //cudaDeviceSynchronize();

        // D→H transfer
        deviceToHost(d_gray, grayMat);

        // Display
        cv::imshow("Grayscale", grayMat);

        if (cv::waitKey(1) == 27)
            break;
    }

    running.store(false);
    tCapture.join();
    tPinned.join();

    // Pulizia
    cudaStreamDestroy(stream);
    cudaFree(d_gray);
    cudaFree(d_grayf);
    cudaFree(d_frame);
    cudaFreeHost(pinnedBuffer[0]->data);
    cudaFreeHost(pinnedBuffer[1]->data);
    delete pinnedBuffer[0];
    delete pinnedBuffer[1];

    return 0;
}