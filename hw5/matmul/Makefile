TARGET=mat_mul_gpu mat_mul_cpu
OBJS=mat_mul_gpu.o mat_mul_cpu.o timers.o
i=10000
j=10000
k=10000

CC=gcc
CFLAGS=-g -O2 -Wall

all: $(TARGET)

mat_mul_gpu:mat_mul_gpu.o timers.o
	$(CC) $(LDFLAGS) mat_mul_gpu.o timers.o -o $@ -lm -lOpenCL

mat_mul_cpu:mat_mul_cpu.o timers.o
	$(CC) $(LDFLAGS) mat_mul_cpu.o timers.o -o $@ -lm -lOpenCL

mat_mul_gpu.o:
	$(CC) $(CFLAGS) -c -o mat_mul_gpu.o mat_mul.c

mat_mul_cpu.o:
	$(CC) $(CFLAGS) -DCPU -c -o mat_mul_cpu.o mat_mul.c

timers.o:
	$(CC) $(CFLAGS) -c -o timers.o timers.c

clean:
	rm -rf $(TARGET) $(OBJS) task*

gpu: mat_mul_gpu
	thorq --add --device gpu ./mat_mul_gpu -i $(i) -j $(j) -k $(k)

cpu: mat_mul_cpu
	thorq --add --device cpu ./mat_mul_cpu -i $(i) -j $(j) -k $(k)

run: $(TARGET)
	thorq --add --device gpu ./mat_mul_gpu -i $(i) -j $(j) -k $(k)
	thorq --add --device cpu ./mat_mul_cpu -i $(i) -j $(j) -k $(k)
