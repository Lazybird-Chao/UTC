mainobj = c-ray_main.o
taskobj = c-ray_task.o


bin = cray
default:$(bin)

C++ = g++
CCFLAGS = -O2  -std=c++11 -fmax-errors=10
NWARNING = -w 

GENCODE_SM20 = -gencode arch=compute_20,code=sm_20
GENCODE_SM30 = -gencode arch=compute_30,code=sm_30
GENCODE_SM35 = -gencode arch=compute_35,code=sm_35

NVCC = nvcc
NVCCFLAGS = -O2  -std=c++11
NVCCFLAGS =  $(GENCODE_SM35) --device-c
NVLINKFLAGS = $(GENCODE_SM35) --device-link

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

c-ray_main.o: c-ray_main.cc task.h typeconfig.h
	$(C++) -o $@ $(CCFLAGS) $(INCLUDE) -c $<
c-ray_task.o: c-ray_task.cc task.h typeconfig.h
	$(C++) -o $@ $(CCFLAGS) $(INCLUDE) -c $<
    
$(bin): $(mainobj) $(taskobj) $(helperobj)
	$(C++) -o $@ $^  $(LIB)

.PHONY: clean
clean:
	rm -f $(mainobj) $(taskobj) $(helperObj) $(bin)
