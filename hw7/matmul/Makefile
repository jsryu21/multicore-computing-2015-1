TARGET=mat_mul
OBJS=mat_mul.o timers.o

CC=mpicc
CFLAGS=-O2 -Wall
LDFLAGS=

all: $(TARGET)

$(TARGET):$(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o $@ -lm

clean:
	rm -rf $(TARGET) $(OBJS) task*

run: $(TARGET)
	thorq --add --mode mpi --nodes 4 --slots 16 ./$(TARGET) -m 8192
