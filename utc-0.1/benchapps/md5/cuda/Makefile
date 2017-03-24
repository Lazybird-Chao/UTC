obj = md5_main.o
bin = md5_main
kernelObj = md5_kernel.o
default: $(bin)

C++ = g++
CCFLAGS = -O2 -std=c++11
NWARNING = -w 

#CUDA_DIR = /usr/local/cuda
CUDA_DIR = /shared/apps/cuda7.5
CUDA_INCLUDE = -I$(CUDA_DIR)/include
CUDA_LIB := -Wl,-rpath,$(CUDA_DIR)/lib64 -L$(CUDA_DIR)/lib64 -lcudart

GENCODE_SM20 = -gencode arch=compute_20,code=sm_20
GENCODE_SM30 = -gencode arch=compute_30,code=sm_30
GENCODE_SM35 = -gencode arch=compute_35,code=sm_35

NVCC = nvcc
NVCCFLAGS = -g -G $(GENCODE_SM35) --device-c
NVLINKFLAGS = $(GENCODE_SM35) --device-link


COMMONDIR = ../../common
helperObj = helperObj.o
$(helperObj):$(COMMONDIR)/helper_getopt.c
	$(C++) -o $@  -c $<

	
md5_main.o: md5_main.cu
	$(NVCC) -o $@   $(CCFLAGS) $(NVCCFLAGS) $(NWARNING)  $(CUDA_INCLUDE) -c $<

md5_kernel.o: md5_kernel.cc
	$(NVCC) -o $@ -x cu  $(CCFLAGS) $(NVCCFLAGS) $(NWARNING)  $(CUDA_INCLUDE) -c $<

tmplink.o: md5_main.o md5_kernel.o
	$(NVCC) -o $@ $(NVLINKFLAGS) $^ 
	
md5_main: $(obj) $(kernelObj) $(helperObj) tmplink.o
	$(C++) -o $@ $^  $(CUDA_LIB)
	

	
.PHONY: clean
clean:
	rm -f $(obj) $(bin) $(kernelObj) $(helperObj) tmplink.o
