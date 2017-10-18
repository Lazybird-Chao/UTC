obj = nbody_main.o bodysystem.o
bin = nbody
default: $(bin)

C++ = g++
CCFLAGS = -O2 -g -std=c++11
LIB = -lm -lrt

DEBUG = 0
ifeq ($(DEBUG), 1)
	CCFLAGS += -g
endif

COMMONDIR = ../../common
helperObj = helperObj.o
$(helperObj):$(COMMONDIR)/helper_getopt.c
	$(C++) -o $@  -c $<
	
%.o:%.cc
	$(C++) -o $@ $(CCFLAGS) -c $<

nbody_main.o: nbody_main.cc nbody.h bodysystem.h

bodysystem.o: bodysystem.cc bodysystem.h nbody.h


$(bin): $(obj) $(helperObj)
	$(C++) -o $@ $(CCFLAGS) $(LIB) $(obj) $(helperObj)

.PHONY: clean
clean:
	rm -f $(obj) $(bin) $(helperObj)
	
	