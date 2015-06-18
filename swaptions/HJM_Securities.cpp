//HJM_Securities.cpp
//Routines to compute various security prices using HJM framework (via Simulation).
//Authors: Mark Broadie, Jatin Dewanwala
//Collaborator: Mikhail Smelyanskiy, Jike Chong, Intel

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <sstream>
#include <sys/time.h>

#include "nr_routines.h"
#include "HJM.h"
#include "HJM_Securities.h"
#include "HJM_type.h"

#ifdef ENABLE_THREAD
#include <pthread.h>
#define MAX_THREAD 1024
#endif
#if defined(ENABLE_CPU) || defined(ENABLE_GPU)
#include <CL/cl.h>
#endif
#ifdef ENABLE_MPI
#include "mpi.h"
#endif

int NUM_TRIALS = DEFAULT_NUM_TRIALS;
int BLOCK_SIZE_ARG = BLOCK_SIZE;
int nThreads = 1;
int local_size = 1;
int nSwaptions = 1;
int iN = 11; 
FTYPE dYears = 5.5; 
int iFactors = 3; 
parm *swaptions;

#if defined(ENABLE_SEQ) || defined(ENABLE_THREAD)
FTYPE** ppdYield;
FTYPE*** pppdFactors;
#elif defined(ENABLE_CPU) || defined(ENABLE_GPU)
FTYPE* pdYield;
FTYPE* pdFactors;
#if defined(ENABLE_GPU)
FTYPE* pdSumSimSwaptionPrice;
FTYPE* pdSumSquareSimSwaptionPrice;
#endif
#endif

static const int MAX_SOURCE_SIZE = 0x100000;

FTYPE *dSumSimSwaptionPrice_global_ptr;
FTYPE *dSumSquareSimSwaptionPrice_global_ptr;

#if defined(ENABLE_SEQ) || defined(ENABLE_THREAD)
void * worker(void *arg){
    int tid = *((int *)arg);
    FTYPE pdSwaptionPrice[2];

    int chunksize = nSwaptions/nThreads;
    int beg = tid*chunksize;
    int end = (tid+1)*chunksize;
    if(tid == nThreads -1)
        end = nSwaptions;

    for(int i=beg; i < end; i++) {
        int iSuccess = HJM_Swaption_Blocking(pdSwaptionPrice,  swaptions[i].dStrike, 
                swaptions[i].dCompounding, swaptions[i].dMaturity, 
                swaptions[i].dTenor, swaptions[i].dPaymentInterval,
                swaptions[i].iN, swaptions[i].iFactors, swaptions[i].dYears, 
                ppdYield[i], pppdFactors[i],
                100, NUM_TRIALS, BLOCK_SIZE, 0);
        assert(iSuccess == 1);
        swaptions[i].dSimSwaptionMeanPrice = pdSwaptionPrice[0];
        swaptions[i].dSimSwaptionStdError = pdSwaptionPrice[1];
    }

    return NULL;    
}
#endif

#if defined(ENABLE_CPU) || defined(ENABLE_GPU)
static void checkErrors(cl_int status, char *label, int line)
{
    switch (status)
    {
        case CL_SUCCESS:
            return;
        case CL_BUILD_PROGRAM_FAILURE:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_BUILD_PROGRAM_FAILURE\n", label, line);
            break;
        case CL_COMPILER_NOT_AVAILABLE:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_COMPILER_NOT_AVAILABLE\n", label, line);
            break;
        case CL_DEVICE_NOT_AVAILABLE:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_DEVICE_NOT_AVAILABLE\n", label, line);
            break;
        case CL_DEVICE_NOT_FOUND:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_DEVICE_NOT_FOUND\n", label, line);
            break;
        case CL_IMAGE_FORMAT_MISMATCH:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_IMAGE_FORMAT_MISMATCH\n", label, line);
            break;
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_IMAGE_FORMAT_NOT_SUPPORTED\n", label, line);
            break;
        case CL_INVALID_ARG_INDEX:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_INVALID_ARG_INDEX\n", label, line);
            break;
        case CL_INVALID_ARG_SIZE:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_INVALID_ARG_SIZE\n", label, line);
            break;
        case CL_INVALID_ARG_VALUE:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_INVALID_ARG_VALUE\n", label, line);
            break;
        case CL_INVALID_BINARY:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_INVALID_BINARY\n", label, line);
            break;
        case CL_INVALID_BUFFER_SIZE:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_INVALID_BUFFER_SIZE\n", label, line);
            break;
        case CL_INVALID_BUILD_OPTIONS:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_INVALID_BUILD_OPTIONS\n", label, line);
            break;
        case CL_INVALID_COMMAND_QUEUE:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_INVALID_COMMAND_QUEUE\n", label, line);
            break;
        case CL_INVALID_CONTEXT:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_INVALID_CONTEXT\n", label, line);
            break;
        case CL_INVALID_DEVICE:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_INVALID_DEVICE\n", label, line);
            break;
        case CL_INVALID_DEVICE_TYPE:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_INVALID_DEVICE_TYPE\n", label, line);
            break;
        case CL_INVALID_EVENT:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_INVALID_EVENT\n", label, line);
            break;
        case CL_INVALID_EVENT_WAIT_LIST:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_INVALID_EVENT_WAIT_LIST\n", label, line);
            break;
        case CL_INVALID_GL_OBJECT:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_INVALID_GL_OBJECT\n", label, line);
            break;
        case CL_INVALID_GLOBAL_OFFSET:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_INVALID_GLOBAL_OFFSET\n", label, line);
            break;
        case CL_INVALID_HOST_PTR:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_INVALID_HOST_PTR\n", label, line);
            break;
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_INVALID_IMAGE_FORMAT_DESCRIPTOR\n", label, line);
            break;
        case CL_INVALID_IMAGE_SIZE:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_INVALID_IMAGE_SIZE\n", label, line);
            break;
        case CL_INVALID_KERNEL_NAME:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_INVALID_KERNEL_NAME\n", label, line);
            break;
        case CL_INVALID_KERNEL:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_INVALID_KERNEL\n", label, line);
            break;
        case CL_INVALID_KERNEL_ARGS:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_INVALID_KERNEL_ARGS\n", label, line);
            break;
        case CL_INVALID_KERNEL_DEFINITION:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_INVALID_KERNEL_DEFINITION\n", label, line);
            break;
        case CL_INVALID_MEM_OBJECT:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_INVALID_MEM_OBJECT\n", label, line);
            break;
        case CL_INVALID_OPERATION:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_INVALID_OPERATION\n", label, line);
            break;
        case CL_INVALID_PLATFORM:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_INVALID_PLATFORM\n", label, line);
            break;
        case CL_INVALID_PROGRAM:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_INVALID_PROGRAM\n", label, line);
            break;
        case CL_INVALID_PROGRAM_EXECUTABLE:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_INVALID_PROGRAM_EXECUTABLE\n", label, line);
            break;
        case CL_INVALID_QUEUE_PROPERTIES:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_INVALID_QUEUE_PROPERTIES\n", label, line);
            break;
        case CL_INVALID_SAMPLER:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_INVALID_SAMPLER\n", label, line);
            break;
        case CL_INVALID_VALUE:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_INVALID_VALUE\n", label, line);
            break;
        case CL_INVALID_WORK_DIMENSION:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_INVALID_WORK_DIMENSION\n", label, line);
            break;
        case CL_INVALID_WORK_GROUP_SIZE:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_INVALID_WORK_GROUP_SIZE\n", label, line);
            break;
        case CL_INVALID_WORK_ITEM_SIZE:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_INVALID_WORK_ITEM_SIZE\n", label, line);
            break;
        case CL_MAP_FAILURE:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_MAP_FAILURE\n", label, line);
            break;
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_MEM_OBJECT_ALLOCATION_FAILURE\n", label, line);
            break;
        case CL_MEM_COPY_OVERLAP:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_MEM_COPY_OVERLAP\n", label, line);
            break;
        case CL_OUT_OF_HOST_MEMORY:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_OUT_OF_HOST_MEMORY\n", label, line);
            break;
        case CL_OUT_OF_RESOURCES:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_OUT_OF_RESOURCES\n", label, line);
            break;
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            fprintf(stderr, "OpenCL error (at %s, line %d): CL_PROFILING_INFO_NOT_AVAILABLE\n", label, line);
            break;
    }
}
#endif

static double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)1.0e-6*tv.tv_usec;
}

//Please note: Whenever we type-cast to (int), we add 0.5 to ensure that the value is rounded to the correct number. 
//For instance, if X/Y = 0.999 then (int) (X/Y) will equal 0 and not 1 (as (int) rounds down).
//Adding 0.5 ensures that this does not happen. Therefore we use (int) (X/Y + 0.5); instead of (int) (X/Y);

int main(int argc, char *argv[])
{
    double start_time = get_time();
    int iSuccess = 0;
    int i,j;

    FTYPE **factors=NULL;
    printf("PARSEC Benchmark Suite\n");
    fflush(NULL);

    if(argc == 1)
    {
        fprintf(stderr," usage: \n\t-ns [number of swaptions (should be > number of threads]\n\t-sm [number of simulations]\n\t-nt [number of threads]\n"); 
        exit(1);
    }

    for (int j=1; j<argc; j++) {
        if (!strcmp("-sm", argv[j])) {NUM_TRIALS = atoi(argv[++j]);}
        else if (!strcmp("-nt", argv[j])) {nThreads = atoi(argv[++j]);}
        else if (!strcmp("-ls", argv[j])) {local_size = atoi(argv[++j]);}
        else if (!strcmp("-ns", argv[j])) {nSwaptions = atoi(argv[++j]);} 
        else {
            fprintf(stderr," usage: \n\t-ns [number of swaptions (should be > number of threads]\n\t-sm [number of simulations]\n\t-nt [number of threads]\n"); 
        }
    }

    printf("Number of Simulations: %d,  Number of threads: %d Number of swaptions: %d\n", NUM_TRIALS, nThreads, nSwaptions);

#if defined(ENABLE_SEQ)
    if (nThreads != 1)
    {
        fprintf(stderr,"Number of threads must be 1 (serial version)\n");
        exit(1);
    }
#elif defined(ENABLE_THREAD)
    pthread_t      *threads;
    pthread_attr_t  pthread_custom_attr;

    if ((nThreads < 1) || (nThreads > MAX_THREAD))
    {
        fprintf(stderr,"Number of threads must be between 1 and %d.\n", MAX_THREAD);
        exit(1);
    }
    threads = (pthread_t *) malloc(nThreads * sizeof(pthread_t));
    pthread_attr_init(&pthread_custom_attr);

    if ((nThreads < 1) || (nThreads > MAX_THREAD))
    {
        fprintf(stderr,"Number of threads must be between 1 and %d.\n", MAX_THREAD);
        exit(1);
    }
#elif defined(ENABLE_CPU) || defined(ENABLE_GPU)
    // CL code
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue command_queue;
    cl_program program;
    cl_mem bufferSwaptions;

    // vector
    cl_mem bufYield;
    cl_mem bufForward;
    cl_mem bufTotalDrift;
    cl_mem bufPayoffDiscountFactors;
    cl_mem bufDiscountingRatePath;
    cl_mem bufSwapRatePath;
    cl_mem bufSwapDiscountFactors;
    cl_mem bufSwapPayoffs;
    cl_mem bufExpRes;
#if defined(ENABLE_GPU)
    cl_mem bufSumSimSwaptionPrice;
    cl_mem bufSumSquareSimSwaptionPrice;
#endif

    // matrix
    cl_mem bufFactors;
    cl_mem bufHJMPath;
    cl_mem bufDrifts;
    cl_mem bufZ;

    cl_kernel kernel;
    size_t sizeSwaptions = nSwaptions * sizeof(parm);

    // vector
    size_t sizeYield = nSwaptions * iN * sizeof(FTYPE);
    size_t sizeForward = nSwaptions * iN * sizeof(FTYPE);
    size_t sizeTotalDrift = nSwaptions * (iN - 1) * sizeof(FTYPE);
    size_t sizePayoffDiscountFactors = nSwaptions * (iN * BLOCK_SIZE_ARG) * sizeof(FTYPE);
    size_t sizeDiscountingRatePath = nSwaptions * (iN * BLOCK_SIZE_ARG) * sizeof(FTYPE);

    FTYPE dMaturity = 1;
    FTYPE ddelt = (FTYPE)(dYears/iN);
    int iSwapVectorLength = (int)(iN - dMaturity / ddelt + 0.5);

    size_t sizeSwapRatePath = nSwaptions * (iSwapVectorLength * BLOCK_SIZE_ARG) * sizeof(FTYPE);
    size_t sizeSwapDiscountFactors = nSwaptions * (iSwapVectorLength * BLOCK_SIZE_ARG) * sizeof(FTYPE);
    size_t sizeSwapPayoffs = nSwaptions * iSwapVectorLength * sizeof(FTYPE);
    size_t sizeExpRes = nSwaptions * ((iN - 1) * BLOCK_SIZE_ARG) * sizeof(FTYPE);

#if defined(ENABLE_GPU)
    size_t sizeSumSimSwaptionPrice = nThreads * sizeof(FTYPE);
    size_t sizeSumSquareSimSwaptionPrice = nThreads * sizeof(FTYPE);
#endif

    // matrix
    size_t sizeFactors = nSwaptions * iFactors * (iN - 1) * sizeof(FTYPE);
    size_t sizeHJMPath = nSwaptions * iN * (iN * BLOCK_SIZE_ARG) * sizeof(FTYPE);
    size_t sizeDrifts = nSwaptions * iFactors * (iN - 1) * sizeof(FTYPE);
    size_t sizeZ = nSwaptions * iFactors * (iN * BLOCK_SIZE_ARG) * sizeof(FTYPE);

    checkErrors(clGetPlatformIDs(1, &platform, NULL), (char*)"clGetPlatformIDs", __LINE__);

#if defined(ENABLE_CPU)
    checkErrors(clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL), (char*)"clGetDeviceIDs", __LINE__);
#elif defined(ENABLE_GPU)
    checkErrors(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL), (char*)"clGetDeviceIDs", __LINE__);
#endif

    int errcode;
    context = clCreateContext(0, 1, &device, NULL, NULL, &errcode);
    checkErrors(errcode, (char*)"clCreateContext", __LINE__);
    command_queue = clCreateCommandQueue(context, device, 0, &errcode);
    checkErrors(errcode, (char*)"clCreateCommandQueue", __LINE__);

#if defined(ENABLE_CPU)
    bufferSwaptions = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeSwaptions, NULL, &errcode);
#elif defined(ENABLE_GPU)
    bufferSwaptions = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeSwaptions, NULL, &errcode);
#endif

    checkErrors(errcode, (char*)"clCreateBuffer", __LINE__);

#if defined(ENABLE_CPU)
    bufYield = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeYield, NULL, &errcode);
#elif defined(ENABLE_GPU)
    bufYield = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeYield, NULL, &errcode);
#endif

    checkErrors(errcode, (char*)"clCreateBuffer", __LINE__);
    bufForward = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeForward, NULL, &errcode);
    checkErrors(errcode, (char*)"clCreateBuffer", __LINE__);
    bufTotalDrift = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeTotalDrift, NULL, &errcode);
    checkErrors(errcode, (char*)"clCreateBuffer", __LINE__);
    bufPayoffDiscountFactors = clCreateBuffer(context, CL_MEM_READ_WRITE, sizePayoffDiscountFactors, NULL, &errcode);
    checkErrors(errcode, (char*)"clCreateBuffer", __LINE__);
    bufDiscountingRatePath = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeDiscountingRatePath, NULL, &errcode);
    checkErrors(errcode, (char*)"clCreateBuffer", __LINE__);
    bufSwapRatePath = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeSwapRatePath, NULL, &errcode);
    checkErrors(errcode, (char*)"clCreateBuffer", __LINE__);
    bufSwapDiscountFactors = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeSwapDiscountFactors, NULL, &errcode);
    checkErrors(errcode, (char*)"clCreateBuffer", __LINE__);
    bufSwapPayoffs = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeSwapPayoffs, NULL, &errcode);
    checkErrors(errcode, (char*)"clCreateBuffer", __LINE__);
    bufExpRes = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeExpRes, NULL, &errcode);
    checkErrors(errcode, (char*)"clCreateBuffer", __LINE__);

#if defined(ENABLE_GPU)
    bufSumSimSwaptionPrice = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeSumSimSwaptionPrice, NULL, &errcode);
    checkErrors(errcode, (char*)"clCreateBuffer", __LINE__);
    bufSumSquareSimSwaptionPrice = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeSumSquareSimSwaptionPrice, NULL, &errcode);
    checkErrors(errcode, (char*)"clCreateBuffer", __LINE__);
#endif

#if defined(ENABLE_CPU)
    bufFactors = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeFactors, NULL, &errcode);
#elif defined(ENABLE_GPU)
    bufFactors = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeFactors, NULL, &errcode);
#endif

    checkErrors(errcode, (char*)"clCreateBuffer", __LINE__);
    bufHJMPath = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeHJMPath, NULL, &errcode);
    checkErrors(errcode, (char*)"clCreateBuffer", __LINE__);
    bufDrifts = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeDrifts, NULL, &errcode);
    checkErrors(errcode, (char*)"clCreateBuffer", __LINE__);
    bufZ = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeZ, NULL, &errcode);
    checkErrors(errcode, (char*)"clCreateBuffer", __LINE__);

    // read kernel code
    FILE* fp;

#if defined(ENABLE_CPU)
    const char fileName[] = "./cpu.cl";
#elif defined(ENABLE_GPU)
    const char fileName[] = "./gpu.cl";
#endif

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

    // create kernel
    program = clCreateProgramWithSource(context, 1, (const char**)&source_str, &source_size, &errcode);
    checkErrors(errcode, (char*)"clCreateProgramWithSource", __LINE__);
    checkErrors(clBuildProgram(program, 1, &device, NULL, NULL, NULL), (char*)"clBuildProgram", __LINE__);

    size_t log_size;
    // First call to know the proper size
    checkErrors(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size), (char*)"clGetProgramBuildInfo", __LINE__);

    if (log_size > 0) {
        char* build_log = new char[log_size + 1];
        // Second call to get the log
        checkErrors(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL), (char*)"clGetProgramBuildInfo", __LINE__);
        build_log[log_size] = '\0';
        printf("%s\n", build_log);
        delete build_log;
    }

    kernel = clCreateKernel(program, "kernel_func", &errcode);
    checkErrors(errcode, (char*)"clCreateKernel", __LINE__);
    checkErrors(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&bufferSwaptions), (char*)"clSetKernelArg", __LINE__);
    checkErrors(clSetKernelArg(kernel, 1, sizeof(cl_int), (void*)&nSwaptions), (char*)"clSetKernelArg", __LINE__);
    checkErrors(clSetKernelArg(kernel, 2, sizeof(cl_int), (void*)&NUM_TRIALS), (char*)"clSetKernelArg", __LINE__);
    checkErrors(clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&BLOCK_SIZE_ARG), (char*)"clSetKernelArg", __LINE__);
    checkErrors(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&bufYield), (char*)"clSetKernelArg", __LINE__);
    checkErrors(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&bufForward), (char*)"clSetKernelArg", __LINE__);
    checkErrors(clSetKernelArg(kernel, 6, sizeof(cl_mem), (void*)&bufTotalDrift), (char*)"clSetKernelArg", __LINE__);
    checkErrors(clSetKernelArg(kernel, 7, sizeof(cl_mem), (void*)&bufPayoffDiscountFactors), (char*)"clSetKernelArg", __LINE__);
    checkErrors(clSetKernelArg(kernel, 8, sizeof(cl_mem), (void*)&bufDiscountingRatePath), (char*)"clSetKernelArg", __LINE__);
    checkErrors(clSetKernelArg(kernel, 9, sizeof(cl_mem), (void*)&bufSwapRatePath), (char*)"clSetKernelArg", __LINE__);
    checkErrors(clSetKernelArg(kernel, 10, sizeof(cl_mem), (void*)&bufSwapDiscountFactors), (char*)"clSetKernelArg", __LINE__);
    checkErrors(clSetKernelArg(kernel, 11, sizeof(cl_mem), (void*)&bufSwapPayoffs), (char*)"clSetKernelArg", __LINE__);
    checkErrors(clSetKernelArg(kernel, 12, sizeof(cl_mem), (void*)&bufExpRes), (char*)"clSetKernelArg", __LINE__);
    checkErrors(clSetKernelArg(kernel, 13, sizeof(cl_mem), (void*)&bufFactors), (char*)"clSetKernelArg", __LINE__);
    checkErrors(clSetKernelArg(kernel, 14, sizeof(cl_mem), (void*)&bufHJMPath), (char*)"clSetKernelArg", __LINE__);
    checkErrors(clSetKernelArg(kernel, 15, sizeof(cl_mem), (void*)&bufDrifts), (char*)"clSetKernelArg", __LINE__);
    checkErrors(clSetKernelArg(kernel, 16, sizeof(cl_mem), (void*)&bufZ), (char*)"clSetKernelArg", __LINE__);

#ifdef ENABLE_GPU
    checkErrors(clSetKernelArg(kernel, 17, sizeof(cl_mem), (void*)&bufSumSimSwaptionPrice), (char*)"clSetKernelArg", __LINE__);
    checkErrors(clSetKernelArg(kernel, 18, sizeof(cl_mem), (void*)&bufSumSquareSimSwaptionPrice), (char*)"clSetKernelArg", __LINE__);
#endif

#endif

    // initialize input dataset
    factors = dmatrix(0, iFactors-1, 0, iN-2);
    //the three rows store vol data for the three factors
    factors[0][0]= .01;
    factors[0][1]= .01;
    factors[0][2]= .01;
    factors[0][3]= .01;
    factors[0][4]= .01;
    factors[0][5]= .01;
    factors[0][6]= .01;
    factors[0][7]= .01;
    factors[0][8]= .01;
    factors[0][9]= .01;

    factors[1][0]= .009048;
    factors[1][1]= .008187;
    factors[1][2]= .007408;
    factors[1][3]= .006703;
    factors[1][4]= .006065;
    factors[1][5]= .005488;
    factors[1][6]= .004966;
    factors[1][7]= .004493;
    factors[1][8]= .004066;
    factors[1][9]= .003679;

    factors[2][0]= .001000;
    factors[2][1]= .000750;
    factors[2][2]= .000500;
    factors[2][3]= .000250;
    factors[2][4]= .000000;
    factors[2][5]= -.000250;
    factors[2][6]= -.000500;
    factors[2][7]= -.000750;
    factors[2][8]= -.001000;
    factors[2][9]= -.001250;

#if defined(ENABLE_SEQ) || defined(ENABLE_THREAD)
    // setting up multiple swaptions
    swaptions = (parm *)malloc(sizeof(parm)*nSwaptions);
    ppdYield = static_cast< FTYPE** >(malloc(nSwaptions * sizeof(FTYPE*)));
    pppdFactors = static_cast< FTYPE*** >(malloc(nSwaptions * sizeof(FTYPE**)));
    for (i = 0; i < nSwaptions; ++i) {
        ppdYield[i] = dvector(0, iN - 1);
        pppdFactors[i] = dmatrix(0, iFactors - 1, 0, iN - 2);
    }
#elif defined(ENABLE_CPU)
    // Enqueue buffer
    swaptions = static_cast< parm* >(clEnqueueMapBuffer(command_queue, bufferSwaptions, CL_FALSE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeSwaptions, 0, NULL, NULL, &errcode));
    checkErrors(errcode, (char*)"clEnqueueMapBuffer", __LINE__);
    pdYield = static_cast< FTYPE* >(clEnqueueMapBuffer(command_queue, bufYield, CL_FALSE, CL_MAP_WRITE, 0, sizeYield, 0, NULL, NULL, &errcode));
    checkErrors(errcode, (char*)"clEnqueueMapBuffer", __LINE__);
    pdFactors = static_cast< FTYPE* >(clEnqueueMapBuffer(command_queue, bufFactors, CL_FALSE, CL_MAP_WRITE, 0, sizeFactors, 0, NULL, NULL, &errcode));
    checkErrors(errcode, (char*)"clEnqueueMapBuffer", __LINE__);
    checkErrors(clFinish(command_queue), (char*)"clFinish", __LINE__);
#elif defined(ENABLE_GPU)
    swaptions = static_cast< parm* >(malloc(sizeSwaptions));
    pdYield = static_cast< FTYPE* >(malloc(sizeYield));
    pdFactors = static_cast< FTYPE* >(malloc(sizeFactors));
#endif

    int k;
    for (i = 0; i < nSwaptions; i++) {
        swaptions[i].Id = i;
        swaptions[i].iN = iN;
        swaptions[i].iFactors = iFactors;
        swaptions[i].dYears = dYears;

        swaptions[i].dStrike =  (double)i / (double)nSwaptions;
        swaptions[i].dCompounding =  0;
        swaptions[i].dMaturity =  1;
        swaptions[i].dTenor =  2.0;
        swaptions[i].dPaymentInterval =  1.0;
#if defined(ENABLE_SEQ) || defined(ENABLE_THREAD)
        ppdYield[i][0] = .1;
        for(j=1;j<=swaptions[i].iN-1;++j)
            ppdYield[i][j] = ppdYield[i][j-1]+.005;

        for(k=0;k<=swaptions[i].iFactors-1;++k)
            for(j=0;j<=swaptions[i].iN-2;++j)
                pppdFactors[i][k][j] = factors[k][j];
#elif defined(ENABLE_CPU) || defined(ENABLE_GPU)
        pdYield[i * iN] = .1;
        for(j=1;j<=swaptions[i].iN-1;++j)
            pdYield[i * iN + j] = pdYield[i * iN + j - 1]+.005;

        for(k=0;k<=swaptions[i].iFactors-1;++k)
            for(j=0;j<=swaptions[i].iN-2;++j)
                pdFactors[i * iFactors * (iN - 1) + k * (iN - 1) + j] = factors[k][j];
#endif
    }

#if defined(ENABLE_GPU)
    checkErrors(clEnqueueWriteBuffer(command_queue, bufferSwaptions, CL_FALSE, 0, sizeSwaptions, swaptions, 0, NULL, NULL), (char*)"clEnqueueWriteBuffer", __LINE__);
    checkErrors(clEnqueueWriteBuffer(command_queue, bufYield, CL_FALSE, 0, sizeYield, pdYield, 0, NULL, NULL), (char*)"clEnqueueWriteBuffer", __LINE__);
    checkErrors(clEnqueueWriteBuffer(command_queue, bufFactors, CL_FALSE, 0, sizeFactors, pdFactors, 0, NULL, NULL), (char*)"clEnqueueWriteBuffer", __LINE__);
    checkErrors(clFinish(command_queue), (char*)"clFinish", __LINE__);
#endif

    // **********Calling the Swaption Pricing Routine*****************
#if defined(ENABLE_SEQ)
    int threadID=0;
    worker(&threadID);
#elif defined(ENABLE_THREAD)
    int threadIDs[nThreads];
    for (i = 0; i < nThreads; i++) {
        threadIDs[i] = i;
        pthread_create(&threads[i], &pthread_custom_attr, worker, &threadIDs[i]);
    }
    for (i = 0; i < nThreads; i++) {
        pthread_join(threads[i], NULL);
    }

    free(threads);
#elif defined(ENABLE_CPU) || defined(ENABLE_GPU)
    // Iterate through number of interations
    size_t global = nThreads;
    size_t local = local_size;
    // launch the kernel
    checkErrors(clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL), (char*)"clEnqueueNDRangeKernel", __LINE__);
    checkErrors(clFinish(command_queue), (char*)"clFinish", __LINE__);
#endif

#if defined(ENABLE_GPU)
    pdSumSimSwaptionPrice = static_cast< FTYPE* >(malloc(sizeSumSimSwaptionPrice));
    pdSumSquareSimSwaptionPrice = static_cast< FTYPE* >(malloc(sizeSumSquareSimSwaptionPrice));
    checkErrors(clEnqueueReadBuffer(command_queue, bufSumSimSwaptionPrice, CL_FALSE, 0, sizeSumSimSwaptionPrice, pdSumSimSwaptionPrice, 0, NULL, NULL), (char*)"clEnqueueReadBuffer", __LINE__);
    checkErrors(clEnqueueReadBuffer(command_queue, bufSumSquareSimSwaptionPrice, CL_FALSE, 0, sizeSumSquareSimSwaptionPrice, pdSumSquareSimSwaptionPrice, 0, NULL, NULL), (char*)"clEnqueueReadBuffer", __LINE__);
    checkErrors(clFinish(command_queue), (char*)"clFinish", __LINE__);

    for (i = 0; i < nSwaptions; ++i) {
        int start_tid = (i * nThreads + nSwaptions - 1) / nSwaptions;
        int end_tid = ((i + 1) * nThreads + nSwaptions - 1) / nSwaptions;
        FTYPE sum = 0.0;
        FTYPE sumSquare = 0.0;
        for (j = start_tid; j < end_tid; ++j) {
            sum += pdSumSimSwaptionPrice[j];
            sumSquare += pdSumSquareSimSwaptionPrice[j];
        }
        printf("%d, sum : %f, sumSquare : %f\n", i, sum, sumSquare);
        swaptions[i].dSimSwaptionMeanPrice = sum / NUM_TRIALS;
        swaptions[i].dSimSwaptionStdError = sqrt((sumSquare-sum*sum/NUM_TRIALS)/
                (NUM_TRIALS-1.0))/sqrt((FTYPE)NUM_TRIALS);
    }

    free(pdSumSimSwaptionPrice);
    free(pdSumSquareSimSwaptionPrice);
#endif

    for (i = 0; i < nSwaptions; i++) {
        fprintf(stderr,"Swaption%d: [SwaptionPrice: %.10lf StdError: %.10lf] \n", 
                i, swaptions[i].dSimSwaptionMeanPrice, swaptions[i].dSimSwaptionStdError);
    }

#if defined(ENABLE_SEQ) || defined(ENABLE_THREAD)
    for (i = 0; i < nSwaptions; i++) {
        free_dvector(ppdYield[i], 0, swaptions[i].iN-1);
        free_dmatrix(pppdFactors[i], 0, swaptions[i].iFactors-1, 0, swaptions[i].iN-2);
    }

    free(pppdFactors);
    free(ppdYield);
#elif defined(ENABLE_CPU) || defined(ENABLE_GPU)
    free(pdFactors);
    free(pdYield);
#endif
    free(swaptions);

    double end_time = get_time();
    printf("Time spent : %lf sec\n", end_time - start_time);

    return iSuccess;
}
