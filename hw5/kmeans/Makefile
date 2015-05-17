
CXX=g++
CXXFLAGS=-Wall


LIBS = -lrt
LDFLAGS = ${LIBS}

all: kmeans_seq kmeans_pthread kmeans_opencl_gpu kmeans_opencl_cpu

kmeans_seq: kmeans_seq.o kmeans_main.o
	${CXX} $^ -o $@ ${LDFLAGS}

kmeans_pthread: kmeans_pthread.o kmeans_main.o
	${CXX} $^ -o $@ ${LDFLAGS} -lpthread

kmeans_opencl_gpu: kmeans_opencl_gpu.o kmeans_main.o
	${CXX} $^ -o $@ ${LDFLAGS} -lOpenCL

kmeans_opencl_cpu: kmeans_opencl_cpu.o kmeans_main.o
	${CXX} $^ -o $@ ${LDFLAGS} -lOpenCL

kmeans_opencl_gpu.o: kmeans_opencl.cpp
	${CXX} ${CXXFLAGS} -c -o kmeans_opencl_gpu.o kmeans_opencl.cpp

kmeans_opencl_cpu.o: kmeans_opencl.cpp
	${CXX} ${CXXFLAGS} -DCPU -c -o kmeans_opencl_cpu.o kmeans_opencl.cpp

run: seq pthread gpu cpu

seq:
	thorq --add kmeans_seq centroid.point data.point result_seq.class final_centroid_seq.point 1024

pthread:
	thorq --add kmeans_pthread centroid.point data.point result_pthread.class final_centroid_pthread.point 1024

gpu:
	thorq --add --device gpu kmeans_opencl_gpu centroid.point data.point result_opencl_gpu.class final_centroid_opencl_gpu.point 1024 $(g)

cpu:
	thorq --add --device cpu kmeans_opencl_cpu centroid.point data.point result_opencl_cpu.class final_centroid_opencl_cpu.point 1024 $(g)

testcase:
	./gen_data.py centroid 16 centroid.point
	./gen_data.py data 1048576 data.point 16

clean:
	rm -f kmeans_seq kmeans_pthread kmeans_opencl_gpu kmeans_opencl_cpu kmeans_main.o kmeans_seq.o kmeans_pthread.o kmeans_opencl_gpu.o kmeans_opencl_cpu.o final_centroid*.point result*.class task* *.png
