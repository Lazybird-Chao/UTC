obj = bfs2_main.o
bin = bfs2_main
kernelObj = bfs2_kernel.o
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

	
bfs2_main.o: bfs2_main.cu bfs2_main.h bfs2_comm_data.h
	$(NVCC) -o $@   $(CCFLAGS) $(NVCCFLAGS) $(NWARNING)  $(CUDA_INCLUDE) -c $<

bfs2_kernel.o: bfs2_kernel.cc bfs2_kernel.h
	$(NVCC) -o $@ -x cu  $(CCFLAGS) $(NVCCFLAGS) $(NWARNING)  $(CUDA_INCLUDE) -c $<

tmplink.o: bfs2_main.o bfs2_kernel.o
	$(NVCC) -o $@ $(NVLINKFLAGS) $^ 
	
bfs2_main: $(obj) $(kernelObj) $(helperObj) tmplink.o
	$(C++) -o $@ $^  $(CUDA_LIB)
	

	
.PHONY: clean
clean:
	rm -f $(obj) $(bin) $(kernelObj) $(helperObj) tmplink.o
