obj = yuvconvert_main.o image.o
bin = yuvconvert_main
default: $(bin)

C++ = g++
CCFLAGS = -O2 -std=c++11
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

yuvconvert_main.o:yuvconvert_main.cc image.h

image.o:image.cc image.h


$(bin): $(obj) $(helperObj)
	$(C++) -o $@ $(CCFLAGS) $(LIB) $(obj) $(helperObj)

.PHONY: clean
clean:
	rm -f $(obj) $(bin) $(helperObj)