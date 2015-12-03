main: main.o cpuUtils.o
	nvcc -o main main.o cpuUtils.o

main.o: main.cu
	nvcc -c main.cu

cpuUtils.o: cpuUtils.cu
	nvcc -c cpuUtils.cu

clear:
	rm *.o main