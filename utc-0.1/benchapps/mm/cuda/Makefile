obj = mm_main.o
bin = mm_main
kernelObj = mm_kernel.o
default: $(bin)

C++ = g++
CCFLAGS = -O2


CUDA_DIR = /usr/local/cuda
CUDA_INCLUDE = -I$(CUDA_DIR)/include
CUDA_LIB := -Wl,-rpath,$(CUDA_DIR)/lib64 -L$(CUDA_DIR)/lib64 -lcudart

GENCODE_SM20 = -gencode arch=compute_20,code=sm_20
GENCODE_SM30 = -gencode arch=compute_30,code=sm_30
GENCODE_SM35 = -gencode arch=compute_35,code=sm_35

NVCC = nvcc
NVCCFLAGS = -O2 $(GENCODE_SM20)



COMMONDIR = ../../common
helperObj = helperObj.o
$(helperObj):$(COMMONDIR)/helper_getopt.c
	$(C++) -o $@  -c $<

	
$(obj):mm_main.cu
	$(NVCC) -o $@ -ccbin g++-4.8 $(CCFLAGS) $(CUDA_INCLUDE) -c $<
#$(C++) -o $@ $(CCFLAGS) $(CUDA_INCLUDE) -c $<

$(kernelObj):mm_kernel.cc
	$(NVCC) -x cu -ccbin g++-4.8 $(NVCCFLAGS) $(CUDA_INCLUDE) -c $<


$(bin): $(obj) $(kernelObj) $(helperObj)
	$(C++) -o $@ $(obj) $(kernelObj) $(helperObj) $(CUDA_LIB)

.PHONY: clean
clean:
	rm -f $(obj) $(bin) $(kernelObj) $(helperObj)
	
	