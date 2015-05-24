TARGET=mat_mul
OBJS=mat_mul.o timers.o

CC=gcc
CFLAGS=-O2 -Wall -fopenmp
LDFLAGS=

all: $(TARGET)

$(TARGET):$(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o $@ -fopenmp -lm

clean:
	rm -rf $(TARGET) $(OBJS) task*

run: $(TARGET)
	thorq --add ./$(TARGET) -t 16 -m 4096
