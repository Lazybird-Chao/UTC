mainobj = 	ryr_main.o image.o
taskobj =	ray_task.o yuv_task.o rotate_task.o output_task.o c-ray_task_sgpu.o
kernelobj = c-ray_kernel.o


bin = ryr
default:$(bin)

C++ = g++
CCFLAGS = -O2  -std=c++11 -fmax-errors=10
NWARNING = -w 

GENCODE_SM20 = -gencode arch=compute_20,code=sm_20
GENCODE_SM30 = -gencode arch=compute_30,code=sm_30
GENCODE_SM35 = -gencode arch=compute_35,code=sm_35

NVCC = nvcc
NVCCFLAGS = -O2  -std=c++11
NVCCFLAGS +=  $(GENCODE_SM30) --device-c
NVLINKFLAGS = $(GENCODE_SM30) --device-link

DEBUG = 0
ifeq ($(DEBUG), 1)
	NVCCFLAGS += -g -G
	CCFLAGS += -g
endif

#####################################################
UTC_HOMEDIR = /home/liu.chao/utc-workingon
# this comm_file define some useful include dir and lib dir:(eg.mpi, boost, cuda)
# this comm_file define some useful include dir and lib dir:(eg.mpi, boost, cuda)
COMM_FILE = $(UTC_HOMEDIR)/for-dis.comm
include $(COMM_FILE)
    
#######################################################

COMMONDIR = $(UTC_HOMEDIR)/benchapps/common
helperObj = helperObj.o
$(helperObj):$(COMMONDIR)/helper_getopt.c
	$(C++) -o $@  -c $<

ryr_main.o: ryr_main.cc common.h task.h cpu_task/ray_task.h cpu_task/yuv_task.h cpu_task/rotate_task.h cpu_task/output_task.h
	$(C++) -o $@ $(CCFLAGS) $(INCLUDE) -c $<
image.o: cpu_task/image.cc cpu_task/image.h
	$(C++) -o $@ $(CCFLAGS) $(INCLUDE) -c $<
ray_task.o: cpu_task/ray_task.cc cpu_task/ray_task.h common.h
	$(C++) -o $@ $(CCFLAGS) $(INCLUDE) -c $<
yuv_task.o: cpu_task/yuv_task.cc cpu_task/yuv_task.h common.h
	$(C++) -o $@ $(CCFLAGS) $(INCLUDE) -c $<
rotate_task.o: cpu_task/rotate_task.cc cpu_task/rotate_task.h cpu_task/image.h 
	$(C++) -o $@ $(CCFLAGS) $(INCLUDE) -c $<
output_task.o: cpu_task/output_task.cc cpu_task/output_task.h cpu_task/image.h
	$(C++) -o $@ $(CCFLAGS) $(INCLUDE) -c $<

c-ray_task_sgpu.o: ./gpu_task/c-ray_task_sgpu.cu ./gpu_task/c-ray_task_sgpu.h ./gpu_task/c-ray_kernel_v2.h
	$(NVCC) -o $@  $(NVCCFLAGS)  $(NWARNING) $(INCLUDE) -c $<
$(kernelobj): ./gpu_task/c-ray_kernel_v2.cc ./gpu_task/c-ray_kernel_v2.h ./gpu_task/c-ray_kernel_device_v2.h
	$(NVCC) -o $@  -x cu $(NVCCFLAGS)  $(NWARNING) $(INCLUDE) -c $<  
tmplink.o: $(taskobj) $(kernelobj)
	$(NVCC) -o $@ $(NVLINKFLAGS) $^

$(bin): $(mainobj) $(taskobj) $(helperobj) $(kernelobj) tmplink.o
	$(C++) -o $@ $(CCFLAGS) $^  $(LIB)

.PHONY: clean
clean:
	rm -f $(mainobj) $(taskobj) $(helperObj) $(bin) $(kernelobj) tmplink.o

