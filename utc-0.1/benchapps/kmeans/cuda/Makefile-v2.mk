obj = kmeans_main.o kmeans_main_v2.o
bin = kmeans_main kmeans_main_v2
kernelObj = kmeans_kernel.o kmeans_kernel_v2.o
default: $(bin)

.PHONY : v1 v2
v1:kmeans_main
v2:kmeans_main_v2

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
	

	
kmeans_main.o:kmeans_main.cu
	$(NVCC) -o $@ -ccbin g++-4.8 $(CCFLAGS) $(CUDA_INCLUDE) -c $<

kmeans_kernel.o:kmeans_kernel.cc
	$(NVCC) -x cu -ccbin g++-4.8 $(NVCCFLAGS) $(CUDA_INCLUDE) -c $<
	
kmeans_main: $(obj) $(kernelObj) $(helperObj)
	$(C++) -o $@ kmeans_main.o kmeans_kernel.o $(helperObj) $(CUDA_LIB)
	
kmeans_main_v2.o:kmeans_main_v2.cu
	$(NVCC) -o $@ -ccbin g++-4.8 $(CCFLAGS) $(CUDA_INCLUDE) -c $<

kmeans_kernel_v2.o:kmeans_kernel_v2.cc
	$(NVCC) -x cu -ccbin g++-4.8 $(NVCCFLAGS) $(CUDA_INCLUDE) -c $<
	
kmeans_main_v2: $(obj) $(kernelObj) $(helperObj)
	$(C++) -o $@ kmeans_main_v2.o kmeans_kernel_v2.o $(helperObj) $(CUDA_LIB)
	
.PHONY: clean
clean:
	rm -f $(obj) $(bin) $(kernelObj) $(helperObj)
	