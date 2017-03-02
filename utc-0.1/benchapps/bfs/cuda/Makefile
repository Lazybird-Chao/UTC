obj = bfs_main.o
bin = bfs_main
kernelObj = bfs_kernel.o
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
NVCCFLAGS = -g -G $(GENCODE_SM30) --device-c
NVLINKFLAGS = $(GENCODE_SM30) --device-link


COMMONDIR = ../../common
helperObj = helperObj.o
$(helperObj):$(COMMONDIR)/helper_getopt.c
	$(C++) -o $@  -c $<

	
bfs_main.o: bfs_main.cu bfs_main.h bfs_comm_data.h
	$(NVCC) -o $@   $(CCFLAGS) $(NVCCFLAGS) $(NWARNING)  $(CUDA_INCLUDE) -c $<

bfs_kernel.o: bfs_kernel.cc bfs_kernel.h
	$(NVCC) -o $@ -x cu  $(CCFLAGS) $(NVCCFLAGS) $(NWARNING)  $(CUDA_INCLUDE) -c $<

tmplink.o: bfs_main.o bfs_kernel.o
	$(NVCC) -o $@ $(NVLINKFLAGS) $^ 
	
bfs_main: $(obj) $(kernelObj) $(helperObj) tmplink.o
	$(C++) -o $@ $^  $(CUDA_LIB)
	

	
.PHONY: clean
clean:
	rm -f $(obj) $(bin) $(kernelObj) $(helperObj) tmplink.o
