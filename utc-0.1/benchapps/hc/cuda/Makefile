obj = hc_main.o hc_main_v2.o
bin = hc_main hc_main_v2
kernelObj = hc_kernel.o
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

	
hc_main.o:hc_main.cu
	$(NVCC) -o $@ -ccbin g++-4.8 $(CCFLAGS) $(CUDA_INCLUDE) -c $<
#$(C++) -o $@ $(CCFLAGS) $(CUDA_INCLUDE) -c $<
hc_main_v2.o:hc_main_v2.cu
	$(NVCC) -o $@ -ccbin g++-4.8 $(CCFLAGS) $(CUDA_INCLUDE) -c $<

$(kernelObj):hc_kernel.cc
	$(NVCC) -x cu -ccbin g++-4.8 $(NVCCFLAGS) $(CUDA_INCLUDE) -c $<


hc_main: $(obj) $(kernelObj) $(helperObj)
	$(C++) -o $@ hc_main.o $(kernelObj) $(helperObj) $(CUDA_LIB)
	
hc_main_v2: $(obj) $(kernelObj) $(helperObj)
	$(C++) -o $@ hc_main_v2.o $(kernelObj) $(helperObj) $(CUDA_LIB)
	

.PHONY: clean
clean:
	rm -f $(obj) $(bin) $(kernelObj) $(helperObj)
	
	