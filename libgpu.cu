#include <string.h>
#include "libgpu.h"

#define CSTMEMSIZE 100
#define MAXPASSSIZE 50
#define MAXCHAR 126
#define MINCHAR 33

__constant__ char devHashes[CSTMEMSIZE];

char* launchKernels(char * gpuHashes, std::vector<Hash> & hashes, const char * format, const int hashSize)
{
	//Get number of hashes
	nHashes = hashes.size();
	//Allocate memory on CPU for storing results
	char * results = NULL;
	results = calloc(MAXPASSSIZE*nHashes, sizeof(char));
	if (!results)
	{
		return NULL;
	}

	//Copy hashes on GPU constant memory
	cudaMemcpyToSymbol(devHashes, gpuHashes, sizeof(char)*hashSize*nHashes);
	//Allocate memory on GPU for storing results
	char * devResults = NULL; 
	cudaMalloc(&devResults, MAXPASSSIZE*nHashes * sizeof(char) ); 

	// launch kernels according to format
	if (strcmp(format, "dummy") == 0)
	{
		bfDummy<<<blocksPerGrid,threadsPerBlock>>>(devResults);
	}


	cudaMemcpy(results, devResults, MAXPASSSIZE*nHashes * sizeof(char),
		cudaMemcpyDeviceToHost);

	cudaFree(devResults);
	return results;
}

__global__ void bfDummy(char * devResults, bool * founded, int nHashes)
{
	//Initialize guess string
	char guess[MAXPASSSIZE];
	char guessHash[4];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	gpuMemset(guess, MAXPASSSIZE, 0);
	incGuess(guess, tid);

	while (!gpuAll(founded, nHashes))
	{
		dummyHashFunc(guess, guessHash);
		for (int i = 0; i < nHashes; ++i)
		{
			if (gpuStrncmp(guessHash, devHashes[i], 4) == 0)
			{
				gpuStrcpy(devResults[i], guess);
				founded[i] = true;
			}
		}
		incGuess(guess, gridDim.x * blockDim.x);
	}
}

__device__ void dummyHashFunc(const char * guess, void * res)
{
	for (int i = 0; i < 4; ++i)
		{
			res[i] = guess[i];
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
	for (int i = 0; i < n; ++i)
	{
		nDiff += (int)(str1[i] != str2[i]);
	}

	return nDiff;
}

__device__ void gpuStrcpy(char * dest, const char * src)
{
	while (*src != 0)
	{
		*(dest++) = *(src++);
	}
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
	if (*c == 0)
	{
		*c = MINCHAR;
		n--;
	}

	if (*c + n > MAXCHAR)
	{
		t = *c;
		*c = MINCHAR;
		return n - MAXCHAR + t;
	}

	*c = *c + n;
	return 0;
}