obj = mm_main.o
bin = mm_main
default: $(bin)

C++ = g++
CCFLAGS = -O2 -std=c++11


COMMONDIR = ../../common
helperObj = helperObj.o
$(helperObj):$(COMMONDIR)/helper_getopt.c
	$(C++) -o $@  -c $<


$(obj):mm_main.cc
	$(C++) -o $@ $(CCFLAGS) -c $<

$(bin): $(obj) $(helperObj)
	$(C++) -o $@ $(CCFLAGS) $(obj) $(helperObj)

.PHONY: clean
clean:
	rm -f $(obj) $(bin) $(helperObj)