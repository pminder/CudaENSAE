#ifndef H_GPULIB
#define H_GPULIB 

#include <vector>
#include "cpuUtils.h"

char * launchKernels(char * gpuHashes, std::vector<Hash> & hashes, const char * format, const int hashSize);

__global__ void bfDummy(char * devResults, bool * founded, int nHashes);
__device__ void dummyHashFunc(const char * guess, char * res);
__device__ void incGuess(char * guess, int N);
__device__ void gpuMemset(char * guess, int n, const char c);
__device__ bool gpuAll(const bool * founded, const int n);
__device__ int  gpuStrncmp(const char * str1, const char * str2, const int n);
__device__ void gpuStrncpy(char * dest, const char * src, const int n);
__device__ int charAddition(char * c, int n);

#endif // H_GPULIB