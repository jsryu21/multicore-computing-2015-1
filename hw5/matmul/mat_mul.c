#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <math.h>
#include "timers.h"
#include <stdbool.h>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define SIZE_I 100
#define SIZE_J 100
#define SIZE_K 100
#define PRECISION 0.00001
#define MAX_SOURCE_SIZE (0x100000)

bool print_matrix = false;
bool validation = false;

/************************** DO NOT TOUCH BELOW HERE ******************************/

void check_mat_mul(float* matrixC, int widthC, int heightC, float* matrixA, int widthA, int heightA, float* matrixB, int widthB, int heightB) {
    int i, j, k;
    float sum;
    bool validated = true;

    printf("Validating the result..\n");

    if (widthA != heightB) {
        validated = false;
    } else if (heightA != heightC) {
        validated = false;
    } else if (widthB != widthC) {
        validated = false;
    } else {
        // C = AB
        for (i = 0; i < heightA; ++i) {
            size_t rowIndexA = (size_t)i * widthA;
            size_t rowIndexC = (size_t)i * widthC;
            for (j = 0; j < widthB; ++j) {
                sum = 0.f;
                for (k = 0; k < widthA; ++k) {
                    size_t indexA = rowIndexA + k;
                    size_t indexB = (size_t)k * widthB + j;
                    sum += matrixA[indexA] * matrixB[indexB];
                }
                size_t indexC = rowIndexC + j;
                float value = matrixC[indexC];
                if (fabs(value - sum) > PRECISION) {
                    printf("c[%d][%d] is differ(value=%lf correct_value=%lf)!!\n", i, j, value, sum);
                    validated = false;
                }
            }
        }
    }

    printf("Validation : ");
    if(validated) {
        printf("SUCCESSFUL.\n");
    } else {
        printf("FAILED.\n");
    }
}

void print_mat(float* matrix, int width, int height) {
    int i, j;
    for (i = 0; i < height; ++i) {
        for (j = 0; j < width; ++j) {
            printf("%8.2lf ", matrix[i * height + j]);
        }
        printf("\n");
    }
}

void print_help(const char* prog_name) {
    printf("Usage: %s [-pvht]\n", prog_name );
    printf("\n");
    printf("OPTIONS\n");
    printf("  -p : print matrix data.\n");
    printf("  -v : validate matrix multiplication.\n");
    printf("  -h : print this page.\n");
}

void parse_opt(int argc, char** argv) {
    int opt;
    while ((opt = getopt(argc, argv, "pvh")) != -1 ) {
        switch(opt) {
            case 'p':
                // print matrix data.
                print_matrix = 1;
                break;

            case 'v':
                // validation
                validation = 1;
                break;

            case 'h':
            default:
                print_help(argv[0]);
                exit(0);
                break;
        }
    }
}

int main(int argc, char** argv) {
    parse_opt(argc, argv);
    // CL code
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue command_queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem bufferA;
    cl_mem bufferB;
    cl_mem bufferC;
    float* hostA;
    float* hostB;
    float* hostC;
    int wA, hA, wB, hB, wC, hC;
    size_t sizeA, sizeB, sizeC;
    hA = hC = SIZE_I;
    wB = wC = SIZE_J;
    wA = hB = SIZE_K;
    sizeA = (size_t)hA * wA * sizeof(float);
    sizeB = (size_t)hB * wB * sizeof(float);
    sizeC = (size_t)hC * wC * sizeof(float);
    hostA = (float*)malloc(sizeA);
    hostB = (float*)malloc(sizeB);
    hostC = (float*)malloc(sizeC);
    // assign matrix
    int i, j;
    for (i = 0; i < hA; ++i) {
        size_t rowIndex = (size_t)wA * i;
        for (j = 0; j < wA; ++j) {
            size_t index = rowIndex + j;
            hostA[index] = 1;
        }
    }
    for (i = 0; i < hB; ++i) {
        size_t rowIndex = (size_t)wB * i;
        for (j = 0; j < wB; ++j) {
            size_t index = rowIndex + j;
            hostB[index] = 2;
        }
    }
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    cl_ulong size;
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &size, 0);
    printf("%d\n", size);
    context = clCreateContext(0, 1, &device, NULL, NULL, NULL);
    command_queue = clCreateCommandQueue(context, device, 0, NULL);
    bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeA, NULL, NULL);
    bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeB, NULL, NULL);
    bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeC, NULL, NULL);
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
    printf("%s\n", source_str);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "matrixmul", NULL);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&bufferC);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&bufferA);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&bufferB);
    clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&wA);
    clSetKernelArg(kernel, 4, sizeof(cl_int), (void*)&wB);
    timer_start(1);
    // prepare the input data
    clEnqueueWriteBuffer(command_queue, bufferA, CL_FALSE, 0, sizeA, hostA, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue, bufferB, CL_FALSE, 0, sizeB, hostB, 0, NULL, NULL);
    size_t global[2] = {wC, hC};
    // launch the kernel
    clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(command_queue, bufferC, CL_TRUE, 0, sizeC, hostC, 0, NULL, NULL);
    clFinish(command_queue);
    timer_stop(1);

    printf("Time elapsed : %lf sec\n", timer_read(1));

    if(validation) {
        check_mat_mul(hostC, wC, hC, hostA, wA, hA, hostB, wB, wB);
    }

    if(print_matrix) {
        printf("MATRIX A: \n");
        print_mat(hostA, wA, hA);

        printf("MATRIX B: \n");
        print_mat(hostB, wB, hB);

        printf("MATRIX C: \n");
        print_mat(hostC, wC, hC);
    }

    free(hostC);
    free(hostB);
    free(hostA);
    return 0;
}
