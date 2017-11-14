obj = ryr_main.o image.o yuv.o rotate.o rotation.o
bin = ryr
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

ryr.o: ryr_main.cc image.h common.h

image.o:image.cc image.h

yuv.o: yuv.cc common.h

rotation.o: rotation.cc rotation.h image.h

rotate.o: rotate.cc common.h image.h rotation.h

$(bin): $(obj) $(helperObj)
	$(C++) -o $@ $(CCFLAGS) $(LIB) $(obj) $(helperObj)

.PHONY: clean
clean:
	rm -f $(obj) $(bin) $(helperObj)
