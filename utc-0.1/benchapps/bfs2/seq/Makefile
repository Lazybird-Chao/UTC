obj = bfs2_main.o
bin = bfs2_main
default: $(bin)

C++ = g++
CCFLAGS = -O2 -std=c++11
LIB = -lrt -lm

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

bfs2_main.o: bfs2_main.cc bfs2_main.h



$(bin): $(obj) $(helperObj)
	$(C++) -o $@ $(CCFLAGS) $(LIB) $(obj) $(helperObj)

.PHONY: clean
clean:
	rm -f $(obj) $(bin) $(helperObj)
	