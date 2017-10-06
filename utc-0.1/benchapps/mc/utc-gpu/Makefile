mainobj = mc_main.o
taskobj = mc_task.o
kernelobj = mc_kernel.o


bin = mc
default:$(bin)

C++ = g++
CCFLAGS = -O2  -std=c++11 -fmax-errors=10
NWARNING = -w 

GENCODE_SM20 = -gencode arch=compute_20,code=sm_20
GENCODE_SM30 = -gencode arch=compute_30,code=sm_30
GENCODE_SM35 = -gencode arch=compute_35,code=sm_35
GENCODE_SM37 = -gencode arch=compute_37,code=sm_37

NVCC = nvcc
NVCCFLAGS = -O2  -std=c++11
NVCCFLAGS +=  $(GENCODE_SM35) --device-c
NVLINKFLAGS = $(GENCODE_SM35) --device-link

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

mc_main.o: mc_main.cc ./gpu/mc_task.h
	$(C++) -o $@ $(CCFLAGS) $(INCLUDE) -c $<
mc_task.o: ./gpu/mc_task.cu ./gpu/mc_task.h ./gpu/mc_kernel.h
	$(NVCC) -o $@ $(NVCCFLAGS) $(NWARNING) $(INCLUDE) -c $<
mc_kernel.o: ./gpu/mc_kernel.cu ./gpu/mc_kernel.h
	$(NVCC) -o $@ $(NVCCFLAGS)  $(NWARNING) $(CUDA_INCLUDE) -c $<
tmplink.o: mc_task.o mc_kernel.o
	$(NVCC) -o $@ $(NVLINKFLAGS) $^
    
$(bin): $(mainobj) $(taskobj) $(kernelobj) tmplink.o $(helperobj)
	$(C++) -o $@ $^  $(LIB)

.PHONY: clean
clean:
	rm -f $(mainobj) $(taskobj) $(kernelobj) $(helperObj) $(bin)
