TARGET=mat_mul
OBJS=mat_mul.o timers.o

CC=mpicc
CFLAGS=-O2 -Wall -fopenmp
LDFLAGS=

all: $(TARGET)

$(TARGET):$(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o $@ -fopenmp -lm

clean:
	rm -rf $(TARGET) $(OBJS) task*

run: $(TARGET)
	thorq --add --mode mpi --nodes 2 --slots 4 ./$(TARGET) -v -m 4096
