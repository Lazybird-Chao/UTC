mainobj = hc_main.o
taskobj = task.o hc_task.o
kernelobj = hc_kernel.o

bin = hc
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

hc_main.o: hc_main.cc task.h ./gpu/hc_task.h
	$(C++) -o $@ $(CCFLAGS) $(INCLUDE) -c $<
task.o: task.cc task.h
	$(C++) -o $@ $(CCFLAGS) $(INCLUDE) -c $<
hc_task.o: ./gpu/hc_task.cu ./gpu/hc_task.h
	$(NVCC) -o $@ $(NVCCFLAGS) $(NWARNING) $(INCLUDE) -c $<
$(kernelobj): ./gpu/hc_kernel.cu ./gpu/hc_kernel.h
	$(NVCC) -o $@  $(NVCCFLAGS) $(NWARNING) $(INCLUDE) -c $<
tmplink.o: $(taskobj) $(kernelobj)
	$(NVCC) -o $@ $(NVLINKFLAGS) $^
    
$(bin): $(mainobj) $(taskobj) $(kernelobj) tmplink.o $(helperobj)
	$(C++) -o $@ $(CCFLAGS) $^  $(LIB)

.PHONY: clean
clean:
	rm -f $(mainobj) $(taskobj) $(kernelobj) tmplink.o $(helperObj) $(bin)
