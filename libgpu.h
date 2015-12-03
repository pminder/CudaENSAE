#ifndef H_GPULIB
#define H_GPULIB 

__global__ void bruteForce(void * hashes, char ** crackedPass, const int size);
__device__ void dummyHash(char * guess, void * res);
__device__ bool hashncmp(void * hash1, void * hash2, const int size);

#endif // H_GPULIB