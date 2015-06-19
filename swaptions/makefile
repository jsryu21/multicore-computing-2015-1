DEF =
INCLUDE =
CXXFLAGS = -O3 -lm -mfma

EXEC = swaptions
EXEC_PTHREAD = swaptions_pthread
EXEC_CPU = swaptions_cpu
EXEC_GPU = swaptions_gpu
EXEC_MPI = swaptions_mpi
EXEC_SNUCL = swaptions_snucl

OBJS= CumNormalInv.o MaxFunction.o RanUnif.o nr_routines.o icdf.o \
	  HJM_SimPath_Forward_Blocking.o HJM.o HJM_Swaption_Blocking.o  \
	  HJM_Securities.o

version=seq

ifdef version
ifeq "$(version)" "seq"
	DEF := $(DEF) -DENABLE_SEQ
endif
ifeq "$(version)" "pthread"
	DEF := $(DEF) -DENABLE_THREAD
	CXXFLAGS := $(CXXFLAGS) -pthread
	EXEC := $(EXEC_PTHREAD)
endif
ifeq "$(version)" "cpu"
	DEF := $(DEF) -DENABLE_CPU
	CXXFLAGS := $(CXXFLAGS) -lOpenCL
	EXEC := $(EXEC_CPU)
endif
ifeq "$(version)" "gpu"
	DEF := $(DEF) -DENABLE_GPU
	CXXFLAGS := $(CXXFLAGS) -lOpenCL
	EXEC := $(EXEC_GPU)
endif
ifeq "$(version)" "mpi"
	CXX = mpic++
	DEF := $(DEF) -DENABLE_GPU -DENABLE_MPI
	CXXFLAGS := $(CXXFLAGS) -lOpenCL
	EXEC := $(EXEC_MPI)
endif
ifeq "$(version)" "snucl"
	CXX = mpic++
	DEF := $(DEF) -DENABLE_GPU
	INCLUDE := $(INCLUDE) -I$(SNUCLROOT)/inc
	LIBS := $(LIBS) -L$(SNUCLROOT)/lib
	LDLIBS := $(LDLIBS) -lsnucl_cluster
	EXEC := $(EXEC_SNUCL)
endif
endif

all: remove_obj_exec $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(DEF) $(OBJS) $(INCLUDE) $(LIBS) $(LDLIBS) -o $(EXEC)

.cpp.o:
	$(CXX) $(CXXFLAGS) $(DEF) $(INCLUDE) $(LIBS) $(LDLIBS) -c $*.cpp -o $*.o

.c.o:
	$(CXX) $(CXXFLAGS) $(DEF) -c $*.c -o $*.o

clean:
	rm -f $(OBJS) $(EXEC) $(EXEC_PTHREAD) $(EXEC_CPU) $(EXEC_GPU) $(EXEC_MPI) $(EXEC_SNUCL) task* gmon.out

remove_obj_exec:
	rm -f $(OBJS) $(EXEC)
