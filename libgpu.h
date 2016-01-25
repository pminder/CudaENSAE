#ifndef H_GPULIB
#define H_GPULIB 

#include <vector>
#include "cpuUtils.h"

//ERROR HANDLING (thanks to NVIDIA Cuda by Example)
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


//Launch kernels in order to brute force a series of hashes
//Args:
//  - gpuHashes: decoded hashes (each hash is exactly hashSize long)
//  - nHashes: number of hashes to brute force
//  - format: hash format (dummy, md5...)
//  - hashSize: length of hashing algorithm signature
//Returns: pointer to array of char contaning results (each result is
//MAXPASSSIZE long)
char * launchKernels(char * gpuHashes, int nHashes, const char * format, const int hashSize, char test);

//Kernels for each supported format. These kernels only try a certain number of
//passwords each time they are called. Otherwise, we had problems like screen
//freezes and our program being interrupted and therefore not working...
//Args:
//  - devResults: where to store results if hash founded
//  - status: first passwords to test
//  - nHashes: number of hashes to brute force
__global__ void bfDummy(char * devResults, char * status, int nHashes);
__global__ void bfMD5(char * devResults, char * status, int nHashes);

//Device functions (that slowly emulate string.h)
__device__ int  gpuStrncmp(const char * str1, const char * str2, const int n);
__device__ void gpuStrncpy(char * dest, const char * src, const int n);
__device__ int devStrlen(const char * s);

//Functions to increment guesses both on CPU (to initialize) and on GPU
__device__ int charAddition(char * c, int n);
__device__ void incGuess(char * guess, int N);
void cpuIncGuess(char * guess, int N);
int cpuCharAddition(char * c, int n);
void initGuesses(char * devStatus, int nThreads);

//MISC
//Compute dummy hash on GPU (simple copy) [MD5 is in md5.h)
__device__ void dummyHashFunc(const char * guess, char * res);
//Returns true if all string in results begin with a non NULL char
bool founded(char * results, int nHashes);

#endif // H_GPULIB
