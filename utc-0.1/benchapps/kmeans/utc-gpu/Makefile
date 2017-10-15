mainobj = kmeans_main.o
taskobj = task.o kmeans_task.o
kernelobj = kmeans_kernel.o


bin = kmeans
default:$(bin)

C++ = g++
CCFLAGS = -O2  -std=c++11 -fmax-errors=10
NWARNING = -w 

GENCODE_SM20 = -gencode arch=compute_20,code=sm_20
GENCODE_SM30 = -gencode arch=compute_30,code=sm_30
GENCODE_SM35 = -gencode arch=compute_35,code=sm_35

NVCC = nvcc
NVCCFLAGS = -O2  -std=c++11
NVCCFLAGS +=  $(GENCODE_SM30) --device-c
NVLINKFLAGS = $(GENCODE_SM30) --device-link

DEBUG = 0
ifeq ($(DEBUG), 1)
	NVCCFLAGS += -g -G
	CCFLAGS += -g
endif

#####################################################
UTC_HOMEDIR = /home/liu.chao/utc-workingon
# this comm_file define some useful include dir and lib dir:(eg.mpi, boost, cuda)
# this comm_file define some useful include dir and lib dir:(eg.mpi, boost, cuda)
COMM_FILE = $(UTC_HOMEDIR)/for-dis.comm
include $(COMM_FILE)
    
#######################################################

COMMONDIR = $(UTC_HOMEDIR)/benchapps/common
helperObj = helperObj.o
$(helperObj):$(COMMONDIR)/helper_getopt.c
	$(C++) -o $@  -c $<

kmeans_main.o: kmeans_main.cc task.h ./gpu/kmeans_task.h
	$(C++) -o $@ $(CCFLAGS) $(INCLUDE) -c $<
task.o: task.cc task.h
	$(C++) -o $@ $(CCFLAGS) $(INCLUDE) -c $<
kmeans_task.o: ./gpu/kmeans_task.cu ./gpu/kmeans_task.h
	$(NVCC) -o $@  $(NVCCFLAGS) $(NWARNING) $(INCLUDE) -c $<
kmeans_kernel.o: ./gpu/kmeans_kernel.cu ./gpu/kmeans_kernel.h
	$(NVCC) -o $@  $(NVCCFLAGS) $(NWARNING) $(INCLUDE) -c $<
tmplink.o: kmeans_task.o kmeans_kernel.o
	$(NVCC) -o $@ $(NVLINKFLAGS) $^
    
$(bin): $(mainobj) $(taskobj) $(kernelobj) tmplink.o $(helperobj)
	$(C++) -o $@ $(CCFLAGS) $^  $(LIB)

.PHONY: clean
clean:
	rm -f $(mainobj) $(taskobj) $(helperObj) $(bin) $(kernelobj) tmplink.o
