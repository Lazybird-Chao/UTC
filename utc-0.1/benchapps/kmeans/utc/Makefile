mainobj = kmeans_main.o
taskobj = kmeans_task_sgpu.o task.o
kernelobj = kmeans_kernel.o

bin = kmeans
default:$(bin)

C++ = g++
CCFLAGS = -O2  -std=c++11
NWARNING = -w 

GENCODE_SM20 = -gencode arch=compute_20,code=sm_20
GENCODE_SM30 = -gencode arch=compute_30,code=sm_30
GENCODE_SM35 = -gencode arch=compute_35,code=sm_35

NVCC = nvcc
NVCCFLAGS = $(GENCODE_SM30) --device-c
NVLINKFLAGS = $(GENCODE_SM30) --device-link

DEBUG = 0
ifeq ($(DEBUG), 1)
	CCFLAGS += -g
	NVCCFLAGS += -g -G
endif

#####################################################
UTC_HOMEDIR = /home/liu.chao/utc-workingon
# this comm_file define some useful include dir and lib dir:(eg.mpi, boost, cuda)
# this comm_file define some useful include dir and lib dir:(eg.mpi, boost, cuda)
COMM_FILE = $(UTC_HOMEDIR)/for-dis.comm
include $(COMM_FILE)
    
#######################################################

COMMONDIR = ../../common
helperObj = helperObj.o
$(helperObj):$(COMMONDIR)/helper_getopt.c
	$(C++) -o $@  -c $<

kmeans_main.o: kmeans_main.cc task.h ./sgpu/kmeans_task_sgpu.h
	$(C++) -o $@ $(CCFLAGS) $(INCLUDE) -c $<
task.o: task.cc task.h
	$(C++) -o $@ $(CCFLAGS) $(INCLUDE) -c $<
    
kmeans_task_sgpu.o: ./sgpu/kmeans_task_sgpu.cu ./sgpu/kmeans_task_sgpu.h ./sgpu/kmeans_kernel.h
	$(NVCC) -o $@  $(CCFLAGS) $(NVCCFLAGS) $(NWARNING) $(INCLUDE) -c $<

$(kernelobj): ./sgpu/kmeans_kernel.cc ./sgpu/kmeans_kernel.h
	$(NVCC) -o $@  -x cu $(CCFLAGS) $(NVCCFLAGS) $(NWARNING) $(INCLUDE) -c $<
    
tmplink.o: kmeans_task_sgpu.o $(kernelobj)
	$(NVCC) -o $@ $(NVLINKFLAGS) $^
    
$(bin): $(mainobj) $(taskobj) $(kernelobj) tmplink.o $(helperObj)
	$(C++) -o $@ $^ $(LIB)

.PHONY: clean
clean:
	rm -f $(mainobj) $(taskobj) $(kernelobj) tmplink.o $(bin)

