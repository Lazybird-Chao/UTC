obj = kmeans_main.o
bin = kmeans_main
kernelObj = kmeans_kernel.o
default: $(bin)


C++ = g++
CCFLAGS = -O2 -std=c++11
#CCFLAGS_W = -Xcompiler -Wno-deprecated,-Wno-write-strings
NWARNING = -w 

# change CUDA_DIR if nessary
CUDA_DIR = /shared/apps/cuda7.5
CUDA_INCLUDE = -I$(CUDA_DIR)/include
CUDA_LIB := -Wl,-rpath,$(CUDA_DIR)/lib64 -L$(CUDA_DIR)/lib64 -lcudart

GENCODE_SM20 = -gencode arch=compute_20,code=sm_20
GENCODE_SM30 = -gencode arch=compute_30,code=sm_30
GENCODE_SM35 = -gencode arch=compute_35,code=sm_35

NVCC = nvcc
NVCCFLAGS = -g  $(GENCODE_SM30) --device-c
NVLINKFLAGS = $(GENCODE_SM30) --device-link


COMMONDIR = ../../common
helperObj = helperObj.o
$(helperObj):$(COMMONDIR)/helper_getopt.c
	$(C++) -o $@  -c $<
	

# the need of "-ccbin" depends on the CUDA SDK version and g++ version
kmeans_main.o:kmeans_main.cu
	$(NVCC) -o $@  $(CCFLAGS) $(NVCCFLAGS) $(NWARNING)  $(CUDA_INCLUDE) -c $<

kmeans_kernel.o:kmeans_kernel.cc
	$(NVCC) -o $@ -x cu $(CCFLAGS) $(NVCCFLAGS) $(NWARNING)  $(CUDA_INCLUDE) -c $<

tmplink.o:kmeans_main.o kmeans_kernel.o
	$(NVCC) -o $@ $(NVLINKFLAGS) $^ 
	
kmeans_main: $(obj) $(kernelObj) $(helperObj) tmplink.o
	$(C++) -o $@ $^  $(CUDA_LIB)
	

	
.PHONY: clean
clean:
	rm -f $(obj) $(bin) $(kernelObj) $(helperObj) tmplink.o
	
