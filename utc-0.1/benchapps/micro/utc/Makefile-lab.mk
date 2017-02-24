mainobj = micro_main.o
taskobj = micro_task.o
kernelobj = micro_kernel.o

bin = microtest
default: $(bin)

C++ = g++
CCFLAGS = -O2  -std=c++11
NWARNING = -w 

#CUDA_DIR = /usr/local/cuda
#CUDA_INCLUDE = -I$(CUDA_DIR)/include
#CUDA_LIB := -Wl,-rpath,$(CUDA_DIR)/lib64 -L$(CUDA_DIR)/lib64 -lcudart

GENCODE_SM20 = -gencode arch=compute_20,code=sm_20
GENCODE_SM30 = -gencode arch=compute_30,code=sm_30
GENCODE_SM35 = -gencode arch=compute_35,code=sm_35

NVCC = nvcc
NVCCFLAGS = -g -G $(GENCODE_SM20) --device-c
NVLINKFLAGS = $(GENCODE_SM20) --device-link


#####################################################
UTC_HOMEDIR = /home/chao/Desktop/utc-0.1
# this comm_file define some useful include dir and lib dir:(eg.mpi, boost, cuda)
COMM_FILE = $(UTC_HOMEDIR)/for-lab.comm
include $(COMM_FILE)
UTC_INCLUDE = -I$(UTC_HOMEDIR)/include \
				-I$(UTC_HOMEDIR)/include/gpu \
				-I$(UTC_HOMEDIR)/include/ScopedData \
				-I$(UTC_HOMEDIR)/include/thread_util
UTC_LIB = $(UTC_HOMEDIR)/lib/libutc.a \
			$(UTC_HOMEDIR)/lib/libutcgpu.a \
			$(UTC_HOMEDIR)/lib/libutc-scopeddata.a
		

# when lib src file change, need rebuild libs
LIBMK_FILE = for-lab.mk
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

$(mainobj): micro_main.cc
	$(C++) -o $@ $(CCFLAGS) $(INCLUDE) -c $<
    
$(taskobj): micro_task.cu
	$(NVCC) -o $@ -ccbin g++-4.8 $(CCFLAGS) $(NVCCFLAGS) $(NWARNING) $(UTC_INCLUDE) $(MPI_INCLUDE) $(CUDA_INCLUDE) $(BOOST_INCLUDE) -c $<

$(kernelobj): micro_kernel.cc
	$(NVCC) -o $@ -x cu -ccbin g++-4.8 $(CCFLAGS) $(NVCCFLAGS) $(NWARNING) $(CUDA_INCLUDE) -c $<
    
tmplink.o: $(taskobj) $(kernelobj)
	$(NVCC) -o $@ $(NVLINKFLAGS) $^
    
$(bin): $(mainobj) $(taskobj) $(kernelobj) tmplink.o
	$(C++) -o $@ $^ $(UTC_LIB) $(MPI_LIB) $(CUDA_LIB) $(BOOST_LIB)

.PHONY: clean
clean:
	rm -f $(mainobj) $(taskobj) $(kernelobj) tmplink.o $(bin)
