mainobj = rotate_main.o image.o
taskobj = rotate_task_sgpu.o
kernelobj = rotate_kernel.o

bin = rotate
default:$(bin)

C++ = g++
CCFLAGS = -O2  -std=c++11
NWARNING = -w 

GENCODE_SM20 = -gencode arch=compute_20,code=sm_20
GENCODE_SM30 = -gencode arch=compute_30,code=sm_30
GENCODE_SM35 = -gencode arch=compute_35,code=sm_35

NVCC = nvcc
NVCCFLAGS = -g -G $(GENCODE_SM35) --device-c
NVLINKFLAGS = $(GENCODE_SM35) --device-link

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

COMMONDIR = ../../../common
helperObj = helperObj.o
$(helperObj):$(COMMONDIR)/helper_getopt.c
	$(C++) -o $@  -c $<

rotate_main.o: rotate_main.cc image.h rotate_task_sgpu.h
	$(C++) -o $@ $(CCFLAGS) $(INCLUDE) -c $<
	
image.o: image.cc image.h
	$(C++) -o $@ $(CCFLAGS) $(INCLUDE) -c $<
    
$(taskobj): rotate_task_sgpu.cu rotate_task_sgpu.h rotate_kernel.h
	$(NVCC) -o $@  $(CCFLAGS) $(NVCCFLAGS) $(NWARNING) $(UTC_INCLUDE) $(MPI_INCLUDE) $(CUDA_INCLUDE) $(BOOST_INCLUDE) -c $<

$(kernelobj): rotate_kernel.cc rotate_kernel.h
	$(NVCC) -o $@ -x cu  $(CCFLAGS) $(NVCCFLAGS) $(NWARNING) $(CUDA_INCLUDE) -c $<
    
tmplink.o: $(taskobj) $(kernelobj)
	$(NVCC) -o $@ $(NVLINKFLAGS) $^
    
$(bin): $(mainobj) $(taskobj) $(kernelobj) tmplink.o
	$(C++) -o $@ $^ $(UTC_LIB) $(MPI_LIB) $(CUDA_LIB) $(BOOST_LIB)

.PHONY: clean
clean:
	rm -f $(mainobj) $(taskobj) $(kernelobj) tmplink.o $(bin)
