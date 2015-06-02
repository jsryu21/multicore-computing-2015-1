#!/bin/bash
./plot_data.py input centroid.point data.point input.png
if [ -f ./final_centroid_pthread.point ]; then
    ./plot_data.py result final_centroid_pthread.point data.point result_pthread.class result_pthread.png
fi
if [ -f ./final_centroid_seq.point ]; then
    ./plot_data.py result final_centroid_seq.point data.point result_seq.class result_seq.png
fi
if [ -f ./final_centroid_opencl_gpu.point ]; then
    ./plot_data.py result final_centroid_opencl_gpu.point data.point result_opencl_gpu.class result_opencl_gpu.png
fi
if [ -f ./final_centroid_opencl_cpu.point ]; then
    ./plot_data.py result final_centroid_opencl_cpu.point data.point result_opencl_cpu.class result_opencl_cpu.png
fi
if [ -f ./final_centroid_openmp.point ]; then
    ./plot_data.py result final_centroid_openmp.point data.point result_openmp.class result_openmp.png
fi
