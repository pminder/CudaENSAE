objects = main.o cpuUtils.o libgpu.o

all: $(objects)
	nvcc -O0 $(objects) -o main

%.o: %.cu
	nvcc -O0 -x cu -I. -dc $< -o $@

clean:
	rm -f *.o main