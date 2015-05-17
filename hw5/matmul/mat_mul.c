#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <math.h>
#include "timers.h"
#include <stdbool.h>
#include <time.h>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define SIZE_I 10000
#define SIZE_J 10000
#define SIZE_K 10000
#define PRECISION 0.00001
#define MAX_SOURCE_SIZE (0x100000)
#define TILE_LEN 5000
#define MAX_BUFFER_SIZE ((size_t)((size_t)TILE_LEN * TILE_LEN * sizeof(float)))

bool print_matrix = false;
bool validation = false;
int size_i;
int size_j;
int size_k;

void check_mat_mul(float* matrixC, float* matrixA, float* matrixB, int size_i, int size_j, int size_k) {
    int i, j, k;
    float sum;
    bool validated = true;

    printf("Validating the result..\n");

    // C = AB
    for (i = 0; i < size_i; ++i) {
        size_t rowIndexA = (size_t)i * size_k;
        size_t rowIndexC = (size_t)i * size_j;
        for (j = 0; j < size_j; ++j) {
            sum = 0.f;
            for (k = 0; k < size_k; ++k) {
                size_t indexA = rowIndexA + k;
                size_t indexB = (size_t)k * size_j + j;
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
            printf("%8.2lf ", matrix[i * width + j]);
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
    while ((opt = getopt(argc, argv, "pvhi:j:k:")) != -1 ) {
        switch(opt) {
            case 'p':
                // print matrix data.
                print_matrix = 1;
                break;

            case 'v':
                // validation
                validation = 1;
                break;
            case 'i':
                size_i = atoi(optarg);
                break;
            case 'j':
                size_j = atoi(optarg);
                break;
            case 'k':
                size_k = atoi(optarg);
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
    srand(time(NULL));
    size_i = SIZE_I;
    size_j = SIZE_J;
    size_k = SIZE_K;
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
    hA = hC = size_i;
    wB = wC = size_j;
    wA = hB = size_k;
    sizeA = (size_t)hA * wA * sizeof(float);
    sizeB = (size_t)hB * wB * sizeof(float);
    sizeC = (size_t)hC * wC * sizeof(float);
#ifndef CPU
    hostA = (float*)malloc(sizeA);
    hostB = (float*)malloc(sizeB);
    hostC = (float*)malloc(sizeC);
    memset(hostA, 0.f, sizeA);
    memset(hostB, 0.f, sizeB);
    memset(hostC, 0.f, sizeC);
    // assign matrix
    int i, j;
    for (i = 0; i < hA; ++i) {
        size_t rowIndex = (size_t)wA * i;
        for (j = 0; j < wA; ++j) {
            size_t index = rowIndex + j;
            hostA[index] = 1;//rand();
        }
    }
    for (i = 0; i < hB; ++i) {
        size_t rowIndex = (size_t)wB * i;
        for (j = 0; j < wB; ++j) {
            size_t index = rowIndex + j;
            hostB[index] = 2;rand();
        }
    }
#endif
    clGetPlatformIDs(1, &platform, NULL);
#ifdef CPU
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
#else
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
#endif
    cl_ulong size;
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &size, 0);
    printf("CL_DEVICE_LOCAL_MEM_SIZE : %lld\n", (long long)size);
    context = clCreateContext(0, 1, &device, NULL, NULL, NULL);
    command_queue = clCreateCommandQueue(context, device, 0, NULL);
#ifdef CPU
    bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeA, NULL, NULL);
    bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeB, NULL, NULL);
    bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeC, NULL, NULL);
#else
    size_t bufferSizeA = MIN(sizeA, MAX_BUFFER_SIZE);
    size_t bufferSizeB = MIN(sizeB, MAX_BUFFER_SIZE);
    size_t bufferSizeC = MIN(sizeC, MAX_BUFFER_SIZE);
    bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, bufferSizeA, NULL, NULL);
    bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, bufferSizeB, NULL, NULL);
    bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bufferSizeC, NULL, NULL);
    float* tileHostA = (float*)malloc(bufferSizeA);
    float* tileHostB = (float*)malloc(bufferSizeB);
    float* tileHostC = (float*)malloc(bufferSizeC);
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
    printf("%s\n", source_str);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "matrixmul", NULL);
#ifdef CPU
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&bufferC);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&bufferA);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&bufferB);
    clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&wA);
    clSetKernelArg(kernel, 4, sizeof(cl_int), (void*)&wB);
#else
    int tileWidthA = MIN(wA, TILE_LEN);
    int tileWidthB = MIN(wB, TILE_LEN);
    int tileWidthC = MIN(wC, TILE_LEN);
    int tileHeightC = MIN(hC, TILE_LEN);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&bufferC);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&bufferA);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&bufferB);
    clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&tileWidthA);
    clSetKernelArg(kernel, 4, sizeof(cl_int), (void*)&tileWidthB);
#endif
    timer_start(1);
#ifdef CPU
    hostA = clEnqueueMapBuffer(command_queue, bufferA, CL_TRUE, CL_MAP_READ, 0, sizeA, 0, NULL, NULL, NULL);
    hostB = clEnqueueMapBuffer(command_queue, bufferB, CL_TRUE, CL_MAP_READ, 0, sizeB, 0, NULL, NULL, NULL);
    hostC = clEnqueueMapBuffer(command_queue, bufferC, CL_TRUE, CL_MAP_WRITE, 0, sizeC, 0, NULL, NULL, NULL);
    memset(hostA, 0.f, sizeA);
    memset(hostB, 0.f, sizeB);
    memset(hostC, 0.f, sizeC);
    // assign matrix
    int i, j;
    for (i = 0; i < hA; ++i) {
        size_t rowIndex = (size_t)wA * i;
        for (j = 0; j < wA; ++j) {
            size_t index = rowIndex + j;
            hostA[index] = 1;//rand();
        }
    }
    for (i = 0; i < hB; ++i) {
        size_t rowIndex = (size_t)wB * i;
        for (j = 0; j < wB; ++j) {
            size_t index = rowIndex + j;
            hostB[index] = 2;rand();
        }
    }
    size_t global[2] = {wC, hC};
    // launch the kernel
    clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global, NULL, 0, NULL, NULL);
#else
    int ii, jj, kk;
    for (ii = 0; ii < size_i; ii += TILE_LEN) {
        int iiPB = MIN(ii + TILE_LEN, size_i);
        int diffI = iiPB - ii;
        for (jj = 0; jj < size_j; jj += TILE_LEN) {
            int jjPB = MIN(jj + TILE_LEN, size_j);
            int diffJ = jjPB - jj;
            for (kk = 0; kk < size_k; kk += TILE_LEN) {
                int kkPB = MIN(kk + TILE_LEN, size_k);
                int diffK = kkPB - kk;
                memset(tileHostA, 0.f, bufferSizeA);
                memset(tileHostB, 0.f, bufferSizeB);
                memset(tileHostC, 0.f, bufferSizeC);
                int p;
                for (p = 0; p < diffI; ++p) {
                    memcpy(&tileHostA[p * tileWidthA], &hostA[(ii + p) * wA + kk], diffK * sizeof(float));
                }
                for (p = 0; p < diffK; ++p) {
                    memcpy(&tileHostB[p * tileWidthB], &hostB[(kk + p) * wB + jj], diffJ * sizeof(float));
                }
                // prepare the input data
                clEnqueueWriteBuffer(command_queue, bufferA, CL_FALSE, 0, bufferSizeA, tileHostA, 0, NULL, NULL);
                clEnqueueWriteBuffer(command_queue, bufferB, CL_FALSE, 0, bufferSizeB, tileHostB, 0, NULL, NULL);
                size_t global[2] = {tileWidthC, tileHeightC};
                // launch the kernel
                clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global, NULL, 0, NULL, NULL);
                clEnqueueReadBuffer(command_queue, bufferC, CL_TRUE, 0, bufferSizeC, tileHostC, 0, NULL, NULL);
                int q, r;
                for (q = 0; q < diffI; ++q) {
                    for (r = 0; r < diffJ; ++r) {
                        float value = tileHostC[q * tileWidthC + r];
                        hostC[(ii + q) * wC + jj + r] += value;
                    }
                }
            }
        }
    }
#endif
    clFinish(command_queue);
    timer_stop(1);
    for (i = 0; i < 100; ++i) {
        size_t r = rand() % ((size_t)wC * hC);
        printf("C[%zu] : %lld\n", r, (long long)hostC[r]);
    }

    printf("Time elapsed : %lf sec\n", timer_read(1));

    if(validation) {
        check_mat_mul(hostC, hostA, hostB, size_i, size_j, size_k);
    }

    if(print_matrix) {
        printf("MATRIX A: \n");
        print_mat(hostA, wA, hA);

        printf("MATRIX B: \n");
        print_mat(hostB, wB, hB);

        printf("MATRIX C: \n");
        print_mat(hostC, wC, hC);
    }

#ifndef CPU
    free(tileHostC);
    free(tileHostB);
    free(tileHostA);
#endif
    free(hostC);
    free(hostB);
    free(hostA);
    return 0;
}
