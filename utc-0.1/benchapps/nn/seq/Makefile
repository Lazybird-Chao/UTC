obj = nn_main.o file_io.o
bin = nn_main
default: $(bin)

C++ = g++
CCFLAGS = -O2  -std=c++11
CCFLAGS += -Wno-deprecated -Wno-write-strings
LIB = -lrt -lm

DEBUG = 0
ifeq ($(DEBUG), 1)
	CCFLAGS += -g
endif

COMMONDIR = ../../common
helperObj = helperObj.o
$(helperObj):$(COMMONDIR)/helper_getopt.c
	$(C++) -o $@  -c $<
  
nn_main.o:nn_main.cc nn.h
	$(C++) -o $@ $(CCFLAGS) -c $<

file_io.o:file_io.cc nn.h
	$(C++) -o $@ $(CCFLAGS) -c $<

$(bin): $(obj) $(helperObj)
	$(C++) -o $@ $(CCFLAGS) $(LIB) $(obj) $(helperObj)

.PHONY: clean
clean:
	rm -f $(obj) $(bin) $(helperObj)
