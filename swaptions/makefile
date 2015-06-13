DEF =
INCLUDE =

EXEC = swaptions
EXEC_PTHREAD = swaptions_pthread
EXEC_CPU = swaptions_cpu
EXEC_GPU = swaptions_gpu
EXEC_MPI = swaptions_mpi
EXEC_SNUCL = swaptions_snucl

OBJS= CumNormalInv.o MaxFunction.o RanUnif.o nr_routines.o icdf.o \
	  HJM_SimPath_Forward_Blocking.o HJM.o HJM_Swaption_Blocking.o  \
	  HJM_Securities.o

ifdef version
ifeq "$(version)" "pthread"
	DEF := $(DEF) -DENABLE_THREADS
	CXXFLAGS := $(CXXFLAGS) -pthread -pg -O3 -lm -mfma
	EXEC := $(EXEC_PTHREAD)
endif
ifeq "$(version)" "cpu"
	DEF := $(DEF) -DENABLE_THREADS
	CXXFLAGS := $(CXXFLAGS) -pthread
	EXEC := $(EXEC_CPU)
endif
ifeq "$(version)" "gpu"
	DEF := $(DEF) -DENABLE_THREADS
	CXXFLAGS := $(CXXFLAGS) -pthread
	EXEC := $(EXEC_GPU)
endif
ifeq "$(version)" "mpi"
	DEF := $(DEF) -DENABLE_THREADS
	CXXFLAGS := $(CXXFLAGS) -pthread
	EXEC := $(EXEC_MPI)
endif
ifeq "$(version)" "snucl"
	DEF := $(DEF) -DENABLE_THREADS
	CXXFLAGS := $(CXXFLAGS) -pthread
	EXEC := $(EXEC_SNUCL)
endif
endif

all: remove_obj_exec $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(DEF) $(OBJS) $(INCLUDE) $(LIBS) -o $(EXEC)

.cpp.o:
	$(CXX) $(CXXFLAGS) $(DEF) -c $*.cpp -o $*.o

.c.o:
	$(CXX) $(CXXFLAGS) $(DEF) -c $*.c -o $*.o

clean:
	rm -f $(OBJS) $(EXEC) $(EXEC_PTHREAD) $(EXEC_CPU) $(EXEC_GPU) $(EXEC_MPI) $(EXEC_SNUCL) task*

remove_obj_exec:
	rm -f $(OBJS) $(EXEC)
