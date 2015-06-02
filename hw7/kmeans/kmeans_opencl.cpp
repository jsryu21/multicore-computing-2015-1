#include "kmeans.h"
#include <vector>
#include <CL/cl.h>
#include <cstdio>
#include <cstring>

static const int MAX_SOURCE_SIZE = 0x100000;

int kmeans(int iteration_n, int class_n, int data_n, Point* centroids, Point* data, int* partitioned, int num_threads, int local_size, int argc, char** argv)
{
    // Count number of data in each class
    std::vector< int > count(class_n);
    // CL code
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue command_queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem bufferCentroids;
    cl_mem bufferData;
    cl_mem bufferPartitioned;
    cl_mem bufferTempCentroids;
    cl_mem bufferTempCount;
    clGetPlatformIDs(1, &platform, NULL);
#ifdef CPU
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
#else
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
#endif
    context = clCreateContext(0, 1, &device, NULL, NULL, NULL);
    command_queue = clCreateCommandQueue(context, device, 0, NULL);
    size_t sizeCentroids = class_n * sizeof(Point);
    size_t sizeData = data_n * sizeof(Point);
    size_t sizePartitioned = data_n * sizeof(int);
    size_t sizeTempCentroids = num_threads * class_n * sizeof(Point);
    size_t sizeTempCount = num_threads * class_n * sizeof(int);
#ifdef CPU
    bufferCentroids = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeCentroids, NULL, NULL);
    bufferData = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeData, NULL, NULL);
    bufferPartitioned = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizePartitioned, NULL, NULL);
    bufferTempCentroids = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeTempCentroids, NULL, NULL);
    bufferTempCount = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeTempCount, NULL, NULL);
#else
    bufferCentroids = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeCentroids, NULL, NULL);
    bufferData = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeData, NULL, NULL);
    bufferPartitioned = clCreateBuffer(context, CL_MEM_READ_WRITE, sizePartitioned, NULL, NULL);
    bufferTempCentroids = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeTempCentroids, NULL, NULL);
    bufferTempCount = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeTempCount, NULL, NULL);
#endif
    FILE* fp;
    const char fileName[] = "./kernel.cl";
    size_t source_size;
    char* source_str;
    fp = fopen(fileName, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);
    program = clCreateProgramWithSource(context, 1, (const char**)&source_str, &source_size, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "dispose_data", NULL);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&bufferCentroids);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&bufferData);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&bufferPartitioned);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&bufferTempCentroids);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&bufferTempCount);
    clSetKernelArg(kernel, 5, sizeof(cl_int), (void*)&data_n);
    clSetKernelArg(kernel, 6, sizeof(cl_int), (void*)&class_n);
    clSetKernelArg(kernel, 7, sizeof(cl_int), (void*)&num_threads);
    // prepare the input data
#ifdef CPU
    Point* hostCentroids = static_cast< Point* >(clEnqueueMapBuffer(command_queue, bufferCentroids, CL_TRUE, CL_MAP_READ, 0, sizeCentroids, 0, NULL, NULL, NULL));
    Point* hostData = static_cast< Point* >(clEnqueueMapBuffer(command_queue, bufferData, CL_TRUE, CL_MAP_READ, 0, sizeData, 0, NULL, NULL, NULL));
    int* hostPartitioned = static_cast< int* >(clEnqueueMapBuffer(command_queue, bufferPartitioned, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizePartitioned, 0, NULL, NULL, NULL));
    Point* hostTempCentroids = static_cast< Point* >(clEnqueueMapBuffer(command_queue, bufferTempCentroids, CL_TRUE, CL_MAP_WRITE, 0, sizeTempCentroids, 0, NULL, NULL, NULL));
    int* hostTempCount = static_cast< int* >(clEnqueueMapBuffer(command_queue, bufferTempCount, CL_TRUE, CL_MAP_WRITE, 0, sizeTempCount, 0, NULL, NULL, NULL));
    memcpy(hostData, data, sizeData);
#else
    Point* tempCentroids = static_cast< Point* >(malloc(sizeTempCentroids));
    int* tempCount = static_cast< int* >(malloc(sizeTempCount));
    clEnqueueWriteBuffer(command_queue, bufferData, CL_FALSE, 0, sizeData, data, 0, NULL, NULL);
#endif
    for (int i = 0; i < iteration_n; i++) {
#ifdef CPU
        memcpy(hostCentroids, centroids, sizeCentroids);
        memset(hostTempCentroids, 0, sizeTempCentroids);
        memset(hostTempCount, 0, sizeTempCount);
#else
        clEnqueueWriteBuffer(command_queue, bufferCentroids, CL_FALSE, 0, sizeCentroids, centroids, 0, NULL, NULL);
        memset(tempCentroids, 0, sizeTempCentroids);
        memset(tempCount, 0, sizeTempCount);
        clEnqueueWriteBuffer(command_queue, bufferTempCentroids, CL_FALSE, 0, sizeTempCentroids, tempCentroids, 0, NULL, NULL);
        clEnqueueWriteBuffer(command_queue, bufferTempCount, CL_FALSE, 0, sizeTempCount, tempCount, 0, NULL, NULL);
#endif
        // Iterate through number of interations
        size_t global = num_threads;
        size_t local = local_size;
        // launch the kernel
        clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
        clFinish(command_queue);
        std::fill(centroids, centroids + class_n, Point(0.f, 0.f));
        std::fill(count.begin(), count.end(), 0);
#ifdef CPU
        // Update step
        // Clear sum buffer and class count
        for (int thread_i = 0; thread_i < num_threads; ++thread_i) {
            for (int class_i = 0; class_i < class_n; ++class_i) {
                centroids[class_i].x += hostTempCentroids[thread_i * class_n + class_i].x;
                centroids[class_i].y += hostTempCentroids[thread_i * class_n + class_i].y;
                count[class_i] += hostTempCount[thread_i * class_n + class_i];
            }
        }
#else
        clEnqueueReadBuffer(command_queue, bufferTempCentroids, CL_FALSE, 0, sizeTempCentroids, tempCentroids, 0, NULL, NULL);
        clEnqueueReadBuffer(command_queue, bufferTempCount, CL_FALSE, 0, sizeTempCount, tempCount, 0, NULL, NULL);
        clFinish(command_queue);
        for (int thread_i = 0; thread_i < num_threads; ++thread_i) {
            for (int class_i = 0; class_i < class_n; ++class_i) {
                centroids[class_i].x += tempCentroids[thread_i * class_n + class_i].x;
                centroids[class_i].y += tempCentroids[thread_i * class_n + class_i].y;
                count[class_i] += tempCount[thread_i * class_n + class_i];
            }
        }
#endif
        // Divide the sum with number of class for mean point
        for (int class_i = 0; class_i < class_n; class_i++) {
            centroids[class_i].x /= count[class_i];
            centroids[class_i].y /= count[class_i];
        }
    }
#ifdef CPU
    memcpy(partitioned, hostPartitioned, sizePartitioned);
    free(hostTempCount);
    free(hostTempCentroids);
    free(hostPartitioned);
    free(hostData);
    free(hostCentroids);
#else
    clEnqueueReadBuffer(command_queue, bufferPartitioned, CL_TRUE, 0, sizePartitioned, partitioned, 0, NULL, NULL);
    free(tempCount);
    free(tempCentroids);
#endif
    return 0;
}
