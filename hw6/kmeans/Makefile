
CXX=g++
CXXFLAGS=-Wall


LIBS = -lrt
LDFLAGS = ${LIBS}

all: kmeans_seq kmeans_pthread kmeans_opencl_gpu kmeans_opencl_cpu kmeans_openmp

kmeans_seq: kmeans_seq.o kmeans_main.o
	${CXX} $^ -o $@ ${LDFLAGS}

kmeans_pthread: kmeans_pthread.o kmeans_main.o
	${CXX} $^ -o $@ ${LDFLAGS} -lpthread

kmeans_opencl_gpu: kmeans_opencl_gpu.o kmeans_main.o
	${CXX} $^ -o $@ ${LDFLAGS} -lOpenCL

kmeans_opencl_cpu: kmeans_opencl_cpu.o kmeans_main.o
	${CXX} $^ -o $@ ${LDFLAGS} -lOpenCL

kmeans_openmp: kmeans_openmp.o kmeans_main.o
	${CXX} $^ -o $@ ${LDFLAGS} -fopenmp

kmeans_opencl_gpu.o: kmeans_opencl.cpp
	${CXX} ${CXXFLAGS} -c -o kmeans_opencl_gpu.o kmeans_opencl.cpp

kmeans_opencl_cpu.o: kmeans_opencl.cpp
	${CXX} ${CXXFLAGS} -DCPU -c -o kmeans_opencl_cpu.o kmeans_opencl.cpp

kmeans_openmp.o: openmp.cpp
	${CXX} ${CXXFLAGS} -fopenmp -o kmeans_openmp.o -c openmp.cpp

run: seq pthread gpu cpu openmp

seq:
	thorq --add kmeans_seq centroid.point data.point result_seq.class final_centroid_seq.point 1024

pthread:
	thorq --add kmeans_pthread centroid.point data.point result_pthread.class final_centroid_pthread.point 1024 16

gpu:
	thorq --add --device gpu kmeans_opencl_gpu centroid.point data.point result_opencl_gpu.class final_centroid_opencl_gpu.point 1024 2048 8

cpu:
	thorq --add --device cpu kmeans_opencl_cpu centroid.point data.point result_opencl_cpu.class final_centroid_opencl_cpu.point 1024 256

openmp:
	thorq --add kmeans_openmp centroid.point data.point result_openmp.class final_centroid_openmp.point 1024 16

testcase:
	./gen_data.py centroid 64 centroid.point
	./gen_data.py data 65536 data.point 64

clean:
	rm -f kmeans_seq kmeans_pthread kmeans_opencl_gpu kmeans_opencl_cpu kmeans_openmp kmeans_*.o final_centroid_*.point result_*.class task_* *.png
