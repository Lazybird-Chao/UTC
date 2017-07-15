mainobj = c-ray_main.o
taskobj = c-ray_task_mgpu.o
kernelobj = c-ray_kernel.o

bin = c-ray
default:$(bin)

C++ = g++
CCFLAGS = -O2  -std=c++11
NWARNING = -w 

GENCODE_SM20 = -gencode arch=compute_20,code=sm_20
GENCODE_SM30 = -gencode arch=compute_30,code=sm_30
GENCODE_SM35 = -gencode arch=compute_35,code=sm_35

NVCC = nvcc
NVCCFLAGS =  $(GENCODE_SM35) --device-c
NVLINKFLAGS = $(GENCODE_SM35) --device-link

DEBUG = 0
ifeq ($(DEBUG), 1)
	NVCCFLAGS += -g -G
	CCFLAGS += -g
endif

#####################################################
UTC_HOMEDIR = /home/liu.chao/utc-0.1
# this comm_file define some useful include dir and lib dir:(eg.mpi, boost, cuda)
COMM_FILE = $(UTC_HOMEDIR)/for-dis.comm
include $(COMM_FILE)
UTC_INCLUDE = -I$(UTC_HOMEDIR)/include \
				-I$(UTC_HOMEDIR)/include/gpu \
				-I$(UTC_HOMEDIR)/include/ScopedData \
				-I$(UTC_HOMEDIR)/include/thread_util
UTC_LIB = $(UTC_HOMEDIR)/lib/libutc.a \
			$(UTC_HOMEDIR)/lib/libutcgpu.a \
			$(UTC_HOMEDIR)/lib/libutc-scopeddata.a
		

# when lib src file change, need rebuild libs
LIBMK_FILE = for-dis.mk
.PHONY: utc_lib
utc_lib:
	cd $(UTC_HOMEDIR);$(MAKE) -f $(LIBMK_FILE)
	cd $(UTC_HOMEDIR)/src/ScopedData;$(MAKE) -f Make.mk
	cd $(UTC_HOMEDIR)/src/gpu;$(MAKE) -f Make.mk
    
#######################################################

COMMONDIR = ../../common
helperObj = helperObj.o
$(helperObj):$(COMMONDIR)/helper_getopt.c
	$(C++) -o $@  -c $<

c-ray_main.o: c-ray_main.cc ./mgpu/c-ray_task_mgpu.h  task.h
	$(C++) -o $@ $(CCFLAGS) $(INCLUDE) -c $<
    
c-ray_task_mgpu.o: ./mgpu/c-ray_task_mgpu.cu ./mgpu/c-ray_task_mgpu.h ./mgpu/c-ray_kernel_v2.h
	$(NVCC) -o $@  $(CCFLAGS) $(NVCCFLAGS)  $(NWARNING) $(UTC_INCLUDE) $(MPI_INCLUDE) $(CUDA_INCLUDE) $(BOOST_INCLUDE) -c $<

$(kernelobj): ./mgpu/c-ray_kernel_v2.cc ./mgpu/c-ray_kernel_v2.h ./mgpu/c-ray_kernel_device_v2.h
	$(NVCC) -o $@  -x cu $(CCFLAGS) $(NVCCFLAGS)  $(NWARNING) $(CUDA_INCLUDE) -c $<
    
tmplink.o: $(taskobj) $(kernelobj)
	$(NVCC) -o $@ $(NVLINKFLAGS) $^
    
$(bin): $(mainobj) $(taskobj) $(kernelobj)  $(helperObj) tmplink.o
	$(C++) -o $@ $^  $(UTC_LIB) $(MPI_LIB) $(CUDA_LIB) $(BOOST_LIB)

.PHONY: clean
clean:
	rm -f $(mainobj) $(taskobj) $(kernelobj) tmplink.o $(bin)
