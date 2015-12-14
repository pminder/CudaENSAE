main: main.o cpuUtils.o libgpu.o
	nvcc -dc main.o cpuUtils.o libgpu.o -o main

main.o: main.cu
	nvcc -dc -c main.cu

cpuUtils.o: cpuUtils.cu
	nvcc -dc -c cpuUtils.cu

libgpu.o: libgpu.cu
	nvcc -dc -c libgpu.cu

clear:
	rm *.o main 