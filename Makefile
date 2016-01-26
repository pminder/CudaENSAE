objects = main.o cpuUtils.o libgpu.o

all: $(objects)
	nvcc $(objects) -o main

%.o: %.cu
	nvcc -x cu -I. -dc $< -o $@

clean:
	rm -f *.o main
