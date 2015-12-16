#include <string.h>
#include <stdio.h>
#include "libgpu.h"

#define CSTMEMSIZE 100
#define MAXPASSSIZE 50
#define MAXCHAR 127
#define MINCHAR 33
#define CHARRANGE (MAXCHAR - MINCHAR)

__constant__ char devHashes[CSTMEMSIZE];

// nombre de threads par bloc
const int threadsPerBlock = 256;
const int blocksPerGrid = 32;


__global__ void bfDummy(char * devResults, bool * founded, int nHashes)
{
	//Initialize guess string
	char guess[MAXPASSSIZE];
	char guessHash[4];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	gpuMemset(guess, MAXPASSSIZE, 0);
	incGuess(guess, tid  + 1);

	while (!gpuAll(founded, nHashes))
	{
		dummyHashFunc(guess, guessHash);
		for (int i = 0; i < nHashes; ++i)
		{
			if (gpuStrncmp(guessHash, devHashes + i*4, 4) == 0)
			{
				printf("%s\n", guess);
				// gpuStrncpy(devResults + i*MAXPASSSIZE, guess, MAXPASSSIZE);
				// printf("%s\n", devResults + i*MAXPASSSIZE);
				founded[i] = true;
			}
		}
		incGuess(guess, gridDim.x * blockDim.x);
	}
}


char* launchKernels(char * gpuHashes, std::vector<Hash> & hashes, const char * format, const int hashSize)
{
	//Get number of hashes
	int nHashes = hashes.size();
	//Allocate memory on CPU for storing results
	char * results = NULL;
	results = (char *)calloc(MAXPASSSIZE*nHashes, sizeof(char));
	if (!results)
	{
		return NULL;
	}

	//Copy hashes on GPU constant memory
	cudaMemcpyToSymbol(devHashes, gpuHashes, sizeof(char)*hashSize*nHashes);
	//Allocate memory on GPU for storing results
	char * devResults = NULL; 
	cudaMalloc((void**)&devResults, MAXPASSSIZE*nHashes*sizeof(char));
	//prepare array of booleans for gpu
	bool* founded;
	cudaMalloc((void**)&founded, nHashes*sizeof(bool));
	cudaMemset ((void *)founded, (int)false, (size_t)nHashes);

	// launch kernels according to format
	if (strcmp(format, "dummy") == 0)
	{
		bfDummy<<<threadsPerBlock,blocksPerGrid>>>(devResults, founded, nHashes);
	}


	cudaMemcpy(results, devResults, MAXPASSSIZE*nHashes*sizeof(char),
		cudaMemcpyDeviceToHost);

	cudaFree(devResults);
	cudaFree(founded);
	return results;
}

__device__ void dummyHashFunc(const char * guess, char * hash)
{
	for (int i = 0; i < 4; ++i)
		{
			hash[i] = guess[i];
		}	
}

__device__ void gpuMemset(char * guess, const int n, const char c)
{
	for (int i = 0; i < n; ++i)
	{
		guess[i] = c;
	}
}

__device__ bool gpuAll(const bool * founded, const int n)
{
	bool output(true);
	for (int i = 0; i < n; ++i)
	{
		output = output & founded[i];
	}

	return output;
}

__device__ int  gpuStrncmp(const char * str1, const char * str2, const int n)
{
	int nDiff(0);
	int i(0);
	while ((i != n) && (nDiff == 0)) {
		nDiff = str1[i] - str2[i];
		i++;
	}
	return nDiff;
}

__device__ void gpuStrncpy(char * dest, const char * src, const int n)
{
	// printf("Entering function...\n");
	for (int i = 0; i < n; ++i)
	{
		dest[i] = 'b';
	}

	dest[n-1] = 0;

}

__device__ void incGuess(char * guess, int N)
//potentiellement, il y a un risque de buffer overflow
//mais cela ne devrait pas arriver avant plusieurs milliards années
{
	int i = 0;
	while (N != 0)
	{
		N = charAddition(&guess[i], N);
		i++;
	}
}

__device__ int charAddition(char * c, int n)
{
	char t;
	
	t = *c == 0 ? -1 : *c - MINCHAR;
	*c = ((t + n) % CHARRANGE) + MINCHAR;

	return (t + n)/CHARRANGE;
}