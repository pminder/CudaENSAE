#include <string.h>
#include <stdio.h>
#include "libgpu.h"

#include "md5.h"

#define CSTMEMSIZE 100
#define MAXPASSSIZE 50
#define MAXCHAR 127
#define MINCHAR 33
#define CHARRANGE (MAXCHAR - MINCHAR)
#define ITPERSTEPS 100

__constant__ char devHashes[CSTMEMSIZE];

// nombre de threads par bloc
const int threadsPerBlock = 256;
const int blocksPerGrid = 32;


__global__ void bfDummy(char * devResults, char * status, int nHashes)
{

	//Initialize guess string
	char guess[MAXPASSSIZE];
	//Initialize guess hash
	char guessHash[4];
	//Compute thread ID
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	//Fill in guess string according to previous launches
	gpuStrncpy(guess, status + tid*MAXPASSSIZE, MAXPASSSIZE);

	//Only perform a certain number of guesses 
	for (int iter = 0; iter < ITPERSTEPS; ++iter)
	{
		//Compute current guess hash
		dummyHashFunc(guess, guessHash);
		//For each password hash to crack
		for (int i = 0; i < nHashes; ++i)
		{
			//If password founded
			if (gpuStrncmp(guessHash, devHashes + i*4, 4) == 0)
			{	
				//Store results in devResults
				gpuStrncpy(devResults + i*MAXPASSSIZE, guess, MAXPASSSIZE);
			}
		}
		//increment guess
		incGuess(guess, gridDim.x * blockDim.x);
	}

	//Store last password guess
	gpuStrncpy(status + tid*MAXPASSSIZE, guess, MAXPASSSIZE);
}

__global__ void bfMD5(char * devResults, char * status, int nHashes)
{

	//Initialize guess string
	char guess[MAXPASSSIZE];
	//Initialize guess hash
	char guessHash[16];
	//Compute thread ID
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	//Fill in guess string according to previous launches
	gpuStrncpy(guess, status + tid*MAXPASSSIZE, MAXPASSSIZE);

	//Only perform a certain number of guesses 
	for (int iter = 0; iter < ITPERSTEPS; ++iter)
	{
		//Compute current guess hash
		md5(guessHash, guess, devStrlen(guess));
		//For each password hash to crack
		for (int i = 0; i < nHashes; ++i)
		{
			//If password founded
			if (gpuStrncmp(guessHash, devHashes + i*16, 16) == 0)
			{	
				//Store results in devResults
				gpuStrncpy(devResults + i*MAXPASSSIZE, guess, MAXPASSSIZE);
			}
		}
		//increment guess
		incGuess(guess, gridDim.x * blockDim.x);
	}

	//Store last password guess
	gpuStrncpy(status + tid*MAXPASSSIZE, guess, MAXPASSSIZE);
}

char * launchKernels(char * gpuHashes, std::vector<Hash> & hashes, const char * format, const int hashSize)
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
	cudaMemcpy(devResults, results, MAXPASSSIZE*nHashes*sizeof(char), cudaMemcpyHostToDevice);
	//Compute total number of threads
	int nThreads = threadsPerBlock * blocksPerGrid;
	//Allocate memory in GPU for storing status
	char * devStatus = NULL;
	cudaMalloc((void**)&devStatus, nThreads * MAXPASSSIZE * sizeof(char));
	//Initialize guesses
	initGuesses(devStatus, nThreads);

	md5_init();
	int nIters = 0;

	while (!founded(results, nHashes))
	{
		nIters++;
		// launch kernels according to format
		if (strcmp(format, "dummy") == 0)
		{
			bfDummy<<<threadsPerBlock,blocksPerGrid>>>(devResults, devStatus, nHashes);
		}

		if (strcmp(format, "md5") == 0)
		{
			bfMD5<<<threadsPerBlock,blocksPerGrid>>>(devResults, devStatus, nHashes);
		}


		cudaMemcpy(results, devResults, MAXPASSSIZE*nHashes*sizeof(char),
			cudaMemcpyDeviceToHost);

	}

	printf("Number of iterations: %d\n", nIters);

	cudaFree(devResults);
	cudaFree(devStatus);

	return results;
}

bool founded(char * results, int nHashes)
{
	for (int i = 0; i < nHashes; ++i)
	{
		if (results[i*MAXPASSSIZE] == 0)
		{
			return false;
		}
	}

	return true;
}

void initGuesses(char * devStatus, int nThreads)
{
	//Create equivalent char array on CPU
	char * temp = (char*)calloc(nThreads * MAXPASSSIZE, sizeof(char));
	if (temp == NULL)
	{
		fprintf(stderr, "Memory Error\n");
		return;
	}

	//Fill in cpu array
	for (int i = 0; i < nThreads; ++i)
	{
		cpuIncGuess(temp + i*MAXPASSSIZE, i);
	}

	//Copy on GPU
	cudaMemcpy(devStatus, temp, nThreads * MAXPASSSIZE,
		cudaMemcpyHostToDevice);

	//Free temp array
	free(temp);
}

__device__ int devStrlen(const char * s)
{
	int i = 0;
	while (s[i] != '\x00') {
		++i;
	}

	return i;
}

__device__ void dummyHashFunc(const char * guess, char * hash)
{
	for (int i = 0; i < 4; ++i)
		{
			hash[i] = guess[i];
		}	
}

// __device__ void gpuMemset(char * guess, const int n, const char c)
// {
// 	for (int i = 0; i < n; ++i)
// 	{
// 		guess[i] = c;
// 	}
// }

// __device__ int gpuAll(const int * founded, const int n)
// {
// 	int output(1);
// 	for (int i = 0; i < n; ++i)
// 	{
// 		output &= founded[i];
// 	}

// 	return output;
// }

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
		dest[i] = src[i];
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

void cpuIncGuess(char * guess, int N)
//potentiellement, il y a un risque de buffer overflow
//mais cela ne devrait pas arriver avant plusieurs milliards années
{
	int i = 0;
	while (N != 0)
	{
		N = cpuCharAddition(&guess[i], N);
		i++;
	}
}

int cpuCharAddition(char * c, int n)
{
	char t;
	
	t = *c == 0 ? -1 : *c - MINCHAR;
	*c = ((t + n) % CHARRANGE) + MINCHAR;

	return (t + n)/CHARRANGE;
}