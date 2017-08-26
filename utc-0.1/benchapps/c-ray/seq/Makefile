obj = c-ray_main.o
bin = c-ray_main
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
	
$(obj):c-ray_main.cc
	$(C++) -o $@ $(CCFLAGS) -c $<

$(bin): $(obj) $(helperObj)
	$(C++) -o $@ $(CCFLAGS) $(LIB) $(obj) $(helperObj)

.PHONY: clean
clean:
	rm -f $(obj) $(bin) $(helperObj)

