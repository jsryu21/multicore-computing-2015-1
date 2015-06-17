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

#ifdef ENABLE_SEQ
#endif
#ifdef ENABLE_THREAD
#include <pthread.h>
#define MAX_THREAD 1024
#endif
#ifdef ENABLE_CPU
#include <CL/cl.h>
#endif
#ifdef ENABLE_GPU
#include <CL/cl.h>
#endif
#ifdef ENABLE_MPI
#include <CL/cl.h>
#include "mpi.h"
#endif
#ifdef ENABLE_SNUCL
#endif

int NUM_TRIALS = DEFAULT_NUM_TRIALS;
int BLOCK_SIZE_ARG = BLOCK_SIZE;
int nThreads = 1;
int nSwaptions = 1;
int iN = 11; 
FTYPE dYears = 5.5; 
int iFactors = 3; 
parm *swaptions;

#if defined(ENABLE_SEQ) || defined(ENABLE_THREAD)
FTYPE** ppdYield;
FTYPE*** pppdFactors;
#endif
#ifdef ENABLE_CPU
FTYPE* pdYield;
FTYPE* pdFactors;
#endif

static const int MAX_SOURCE_SIZE = 0x100000;

FTYPE *dSumSimSwaptionPrice_global_ptr;
FTYPE *dSumSquareSimSwaptionPrice_global_ptr;
int chunksize;

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

#if defined(ENABLE_CPU) || defined(ENABLE_GPU) || defined(ENABLE_MPI)
const char *getErrorString(cl_int error)
{
    switch(error){
        // run-time and JIT compiler errors
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

                  // compile-time errors
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

                  // extension errors
        case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        default: return "Unknown OpenCL error";
    }
}

void PrintIfErrors(const std::string& err_message, const cl_int errcode) {
    if (errcode != 0) {
        std::stringstream ss;
        ss << err_message << " " << getErrorString(errcode) << "\n";
        printf(ss.str().c_str());
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
        else if (!strcmp("-ns", argv[j])) {nSwaptions = atoi(argv[++j]);} 
        else {
            fprintf(stderr," usage: \n\t-ns [number of swaptions (should be > number of threads]\n\t-sm [number of simulations]\n\t-nt [number of threads]\n"); 
        }
    }

    if(nSwaptions < nThreads) {
        nSwaptions = nThreads; 
    }

    printf("Number of Simulations: %d,  Number of threads: %d Number of swaptions: %d\n", NUM_TRIALS, nThreads, nSwaptions);

#ifdef ENABLE_SEQ
    if (nThreads != 1)
    {
        fprintf(stderr,"Number of threads must be 1 (serial version)\n");
        exit(1);
    }
#endif
#ifdef ENABLE_THREAD
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
#endif
#ifdef ENABLE_CPU
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

    // matrix
    cl_mem bufFactors;
    cl_mem bufHJMPath;
    cl_mem bufDrifts;
    cl_mem bufZ;
    cl_mem bufRandZ;

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

    // matrix
    size_t sizeFactors = nSwaptions * iFactors * (iN - 1) * sizeof(FTYPE);
    size_t sizeHJMPath = nSwaptions * iN * (iN * BLOCK_SIZE_ARG) * sizeof(FTYPE);
    size_t sizeDrifts = nSwaptions * iFactors * (iN - 1) * sizeof(FTYPE);
    size_t sizeZ = nSwaptions * iFactors * (iN * BLOCK_SIZE_ARG) * sizeof(FTYPE);
    size_t sizeRandZ = nSwaptions * iFactors * (iN * BLOCK_SIZE_ARG) * sizeof(FTYPE);

    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
    context = clCreateContext(0, 1, &device, NULL, NULL, NULL);
    command_queue = clCreateCommandQueue(context, device, 0, NULL);
    bufferSwaptions = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeSwaptions, NULL, NULL);
    bufYield = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeYield, NULL, NULL);
    bufForward = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeForward, NULL, NULL);
    bufTotalDrift = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeTotalDrift, NULL, NULL);
    bufPayoffDiscountFactors = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizePayoffDiscountFactors, NULL, NULL);
    bufDiscountingRatePath = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeDiscountingRatePath, NULL, NULL);
    bufSwapRatePath = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeSwapRatePath, NULL, NULL);
    bufSwapDiscountFactors = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeSwapDiscountFactors, NULL, NULL);
    bufSwapPayoffs = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeSwapPayoffs, NULL, NULL);
    bufExpRes = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeExpRes, NULL, NULL);
    bufFactors = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeFactors, NULL, NULL);
    bufHJMPath = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeHJMPath, NULL, NULL);
    bufDrifts = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeDrifts, NULL, NULL);
    bufZ = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeZ, NULL, NULL);
    bufRandZ = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeRandZ, NULL, NULL);

    // read kernel code
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

    // create kernel
    int errcode;
    program = clCreateProgramWithSource(context, 1, (const char**)&source_str, &source_size, &errcode);
    PrintIfErrors("clCreateProgramWithSource", errcode);
    errcode = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    PrintIfErrors("clBuildProgram", errcode);

    size_t log_size;
    // First call to know the proper size
    errcode = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    PrintIfErrors("clGetProgramBuildInfo", errcode);

    if (log_size > 0) {
        char* build_log = new char[log_size + 1];
        // Second call to get the log
        errcode = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
        build_log[log_size] = '\0';
        printf("%s\n", build_log);
        delete build_log;
    }

    kernel = clCreateKernel(program, "kernel_func", &errcode);
    PrintIfErrors("clCreateKernel", errcode);
    errcode = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&bufferSwaptions);
    PrintIfErrors("clSetKernelArg", errcode);
    errcode = clSetKernelArg(kernel, 1, sizeof(cl_int), (void*)&nSwaptions);
    PrintIfErrors("clSetKernelArg", errcode);
    errcode = clSetKernelArg(kernel, 2, sizeof(cl_int), (void*)&NUM_TRIALS);
    PrintIfErrors("clSetKernelArg", errcode);
    errcode = clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&BLOCK_SIZE_ARG);
    PrintIfErrors("clSetKernelArg", errcode);
    errcode = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&bufYield);
    PrintIfErrors("clSetKernelArg", errcode);
    errcode = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&bufForward);
    PrintIfErrors("clSetKernelArg", errcode);
    errcode = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void*)&bufTotalDrift);
    PrintIfErrors("clSetKernelArg", errcode);
    errcode = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void*)&bufPayoffDiscountFactors);
    PrintIfErrors("clSetKernelArg", errcode);
    errcode = clSetKernelArg(kernel, 8, sizeof(cl_mem), (void*)&bufDiscountingRatePath);
    PrintIfErrors("clSetKernelArg", errcode);
    errcode = clSetKernelArg(kernel, 9, sizeof(cl_mem), (void*)&bufSwapRatePath);
    PrintIfErrors("clSetKernelArg", errcode);
    errcode = clSetKernelArg(kernel, 10, sizeof(cl_mem), (void*)&bufSwapDiscountFactors);
    PrintIfErrors("clSetKernelArg", errcode);
    errcode = clSetKernelArg(kernel, 11, sizeof(cl_mem), (void*)&bufSwapPayoffs);
    PrintIfErrors("clSetKernelArg", errcode);
    errcode = clSetKernelArg(kernel, 12, sizeof(cl_mem), (void*)&bufExpRes);
    PrintIfErrors("clSetKernelArg", errcode);
    errcode = clSetKernelArg(kernel, 13, sizeof(cl_mem), (void*)&bufFactors);
    PrintIfErrors("clSetKernelArg", errcode);
    errcode = clSetKernelArg(kernel, 14, sizeof(cl_mem), (void*)&bufHJMPath);
    PrintIfErrors("clSetKernelArg", errcode);
    errcode = clSetKernelArg(kernel, 15, sizeof(cl_mem), (void*)&bufDrifts);
    PrintIfErrors("clSetKernelArg", errcode);
    errcode = clSetKernelArg(kernel, 16, sizeof(cl_mem), (void*)&bufZ);
    PrintIfErrors("clSetKernelArg", errcode);
    errcode = clSetKernelArg(kernel, 17, sizeof(cl_mem), (void*)&bufRandZ);
    PrintIfErrors("clSetKernelArg", errcode);
#endif
#ifdef ENABLE_GPU
#endif
#ifdef ENABLE_MPI
#endif
#ifdef ENABLE_SNUCL
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
#endif
#ifdef ENABLE_CPU
    // Enqueue buffer
    swaptions = static_cast< parm* >(clEnqueueMapBuffer(command_queue, bufferSwaptions, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeSwaptions, 0, NULL, NULL, &errcode));
    PrintIfErrors("clEnqueueMapBuffer", errcode);
    pdYield = static_cast< FTYPE* >(clEnqueueMapBuffer(command_queue, bufYield, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeYield, 0, NULL, NULL, &errcode));
    PrintIfErrors("clEnqueueMapBuffer", errcode);
    pdFactors = static_cast< FTYPE* >(clEnqueueMapBuffer(command_queue, bufFactors, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeFactors, 0, NULL, NULL, &errcode));
    PrintIfErrors("clEnqueueMapBuffer", errcode);
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
#endif
#ifdef ENABLE_CPU
        pdYield[i * iN] = .1;
        for(j=1;j<=swaptions[i].iN-1;++j)
            pdYield[i * iN + j] = pdYield[i * iN + j - 1]+.005;

        for(k=0;k<=swaptions[i].iFactors-1;++k)
            for(j=0;j<=swaptions[i].iN-2;++j)
                pdFactors[i * iFactors * (iN - 1) + k * (iN - 1) + j] = factors[k][j];
#endif
        printf("host - Id %d, iN %d, iFactors %d, dYears %f, dStrike %f, dCompounding %f, dMaturity %f, dTenor %f, dPaymentInterval %f\n", swaptions[i].Id, swaptions[i].iN, swaptions[i].iFactors, swaptions[i].dYears, swaptions[i].dStrike, swaptions[i].dCompounding, swaptions[i].dMaturity, swaptions[i].dTenor, swaptions[i].dPaymentInterval);
    }

    // **********Calling the Swaption Pricing Routine*****************
#ifdef ENABLE_SEQ
    int threadID=0;
    worker(&threadID);
#endif
#ifdef ENABLE_THREAD
    int threadIDs[nThreads];
    for (i = 0; i < nThreads; i++) {
        threadIDs[i] = i;
        pthread_create(&threads[i], &pthread_custom_attr, worker, &threadIDs[i]);
    }
    for (i = 0; i < nThreads; i++) {
        pthread_join(threads[i], NULL);
    }

    free(threads);
#endif
#ifdef ENABLE_CPU
    // Iterate through number of interations
    size_t global = nThreads;
    size_t local = 16;
    // launch the kernel
    errcode = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    PrintIfErrors("clEnqueueNDRangeKernel", errcode);
    errcode = clFinish(command_queue);
    PrintIfErrors("clFinish", errcode);
#endif
#ifdef ENABLE_GPU
#endif
#ifdef ENABLE_MPI
#endif
#ifdef ENABLE_SNUCL
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
#endif
#ifdef ENABLE_CPU
    free(pdFactors);
    free(pdYield);
#endif
    free(swaptions);

    double end_time = get_time();
    printf("Time spent : %lf sec\n", end_time - start_time);

    return iSuccess;
}
