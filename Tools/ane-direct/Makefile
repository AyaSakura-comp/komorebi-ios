CFLAGS = -O2 -fobjc-arc -Wall -Wno-unused-function
FW = -framework Foundation -framework IOSurface -framework Accelerate -ldl
RT = ../../Sources/KomorebiANEDirect
probe: probe.m $(RT)/ane_rt.m $(RT)/include/ane_rt.h
	xcrun clang $(CFLAGS) -I$(RT)/include -o $@ probe.m $(RT)/ane_rt.m $(FW)
clean:
	rm -f probe
