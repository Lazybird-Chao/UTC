obj = yuvconvert_main.o image.o
bin = yuvconvert_main
kernelObj = yuvconvert_kernel.o
default: $(bin)



C++ = g++
CCFLAGS = -O2  #-std=c++11

# change CUDA_DIR if nessary
CUDA_DIR = /usr/local/cuda
CUDA_INCLUDE = -I$(CUDA_DIR)/include
CUDA_LIB := -Wl,-rpath,$(CUDA_DIR)/lib64 -L$(CUDA_DIR)/lib64 -lcudart

GENCODE_SM20 = -gencode arch=compute_20,code=sm_20
GENCODE_SM30 = -gencode arch=compute_30,code=sm_30
GENCODE_SM35 = -gencode arch=compute_35,code=sm_35

NVCC = nvcc
NVCCFLAGS = -g -G $(GENCODE_SM20) --device-c



COMMONDIR = ../../common
helperObj = helperObj.o
$(helperObj):$(COMMONDIR)/helper_getopt.c
	$(C++) -o $@  -c $<

yuvconvert_main.o: yuvconvert_main.cu image.h yuvconvert_kernel.h
	$(NVCC) -o $@ -ccbin g++-4.8 $(CCFLAGS) $(NVCCFLAGS) $(CUDA_INCLUDE) -c $<

yuvconvert_kernel.o: yuvconvert_kernel.cc yuvconvert_kernel.h image.h
	$(NVCC) -o $@ -x cu -ccbin g++-4.8 $(CCFLAGS) $(NVCCFLAGS) $(CUDA_INCLUDE) -c $<
	
tmplink.o: yuvconvert_main.o yuvconvert_kernel.o
	$(NVCC) -o $@ $(GENCODE_SM20) --device-link $^
	
image.o: image.cc image.h
	$(C++) -o $@  $(CCFLAGS) -c $<
	
yuvconvert_main:$(obj) $(kernelObj) $(helperObj) tmplink.o
	$(C++) -o $@ $^ $(CUDA_LIB)

.PHONY: clean
clean:
	rm -f $(obj) $(bin) $(kernelObj) $(helperObj) tmplink.o

	
