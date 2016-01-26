#include <string.h>
#include <iostream>
#include <stdio.h>
#include "libgpu.h"

#define CSTMEMSIZE 100
#define MAXPASSSIZE 9
#define MAXCHAR 127
#define MINCHAR 33
#define CHARRANGE (MAXCHAR - MINCHAR)
#define ITPERSTEPS 500

// nombre de threads par bloc
#define threadsPerBlock 256
#define blocksPerGrid 64

#include "md5.h"

using namespace std;

__constant__ char devHashes[CSTMEMSIZE];



__global__ void bfDummy(char * devResults, char * status, int nHashes)
{

	//Compute thread ID
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	//Initialize pointer to guess
    //No bank conflicts because MAXPASSSIZE = 9, so using shared memory
    //really is fast ;)
    __shared__ char sharedGuess[threadsPerBlock * MAXPASSSIZE];
    char * guess = sharedGuess + threadIdx.x*MAXPASSSIZE;
	//Initialize to guess hash 
    char guessHash[4];
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

	//Compute thread ID
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	//Initialize pointer to guess
    //No bank conflicts because MAXPASSSIZE = 9, so using shared memory
    //really is fast ;)
	__shared__ char sharedGuess[threadsPerBlock * MAXPASSSIZE];
    char * guess = sharedGuess + threadIdx.x*MAXPASSSIZE;
	//Initialize guess hash
    //As far as compute 2.0 is concerned, there is also local cache, so let's
    //use it too
    char guessHash[17];
    //Initialize normalized guess
    char normalizedGuess[65];
    //Set all values to zero
    for (int i = 0; i < 65; ++i) {
        normalizedGuess[i] = 0;
    }
	//Fill in guess string according to previous launches
	gpuStrncpy(guess, status + tid*MAXPASSSIZE, MAXPASSSIZE);

	//Only perform a certain number of guesses 
	for (int iter = 0; iter < ITPERSTEPS; ++iter)
	{
		//Compute current guess hash
		md5(guessHash, guess, normalizedGuess, devStrlen(guess));
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

char * launchKernels(char * gpuHashes, int nHashes, const char * format, const
        int hashSize, char test)
{
    //Initialize cuda events (for measuring performances)
    cudaEvent_t start, stop;
	//Allocate memory on CPU for storing results
	char * results = NULL;
	results = (char *)calloc(MAXPASSSIZE*nHashes, sizeof(char));
	if (!results)
	{
        cerr << "Memory error" << endl;
		return NULL;
	}

	//Copy hashes on GPU constant memory
	HANDLE_ERROR( cudaMemcpyToSymbol(devHashes, gpuHashes,
                sizeof(char)*hashSize*nHashes) );
	//Allocate memory on GPU for storing results
	char * devResults = NULL; 
	HANDLE_ERROR( cudaMalloc((void**)&devResults,
                MAXPASSSIZE*nHashes*sizeof(char)) );
	HANDLE_ERROR( cudaMemcpy(devResults, results,
                MAXPASSSIZE*nHashes*sizeof(char), cudaMemcpyHostToDevice) );
	//Compute total number of threads
	int nThreads = threadsPerBlock * blocksPerGrid;
	//Allocate memory in GPU for storing status
	char * devStatus = NULL;
	HANDLE_ERROR( cudaMalloc((void**)&devStatus, nThreads * MAXPASSSIZE *
                sizeof(char)) );
	//Initialize guesses on GPU
	initGuesses(devStatus, nThreads);

    //If we want to measure performance
    if (test == 1) {
        HANDLE_ERROR( cudaEventCreate(&start) );
        HANDLE_ERROR( cudaEventCreate(&stop) );
        HANDLE_ERROR( cudaEventRecord(start, 0) );
    }

    //While all passwords are not founded
	while (!founded(results, nHashes))
	{
        //Record start event
        if (test == 1) {
            HANDLE_ERROR( cudaEventRecord(start, 0) );
        }

		// launch kernels according to format
		if (strcmp(format, "dummy") == 0)
		{
			bfDummy<<<threadsPerBlock,blocksPerGrid>>>(devResults, devStatus, nHashes);
		}

		if (strcmp(format, "md5") == 0)
		{
			bfMD5<<<threadsPerBlock,blocksPerGrid>>>(devResults, devStatus, nHashes);
		}

        //Display performances measures
        if (test == 1) {
            HANDLE_ERROR( cudaEventRecord(stop, 0) );
            HANDLE_ERROR( cudaEventSynchronize(stop) );

            //Display number of hashes per second
            float elapsedTime(0);
            float testedHashes = ITPERSTEPS*threadsPerBlock*blocksPerGrid;
            HANDLE_ERROR( cudaEventElapsedTime(&elapsedTime, start, stop) );
            
            cout << "Performances: " << testedHashes / elapsedTime << " hash/ms";
            cout << endl;
        }

        //Copy results from GPU back to CPU
		HANDLE_ERROR( cudaMemcpy(results, devResults,
                    MAXPASSSIZE*nHashes*sizeof(char), cudaMemcpyDeviceToHost) );


	}

    //Environmental concern...
	HANDLE_ERROR( cudaFree(devResults) );
	HANDLE_ERROR( cudaFree(devStatus) );
    if (test == 1) {
        HANDLE_ERROR( cudaEventDestroy(start) );
        HANDLE_ERROR( cudaEventDestroy(stop) );
    }


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
	HANDLE_ERROR( cudaMemcpy(devStatus, temp, nThreads * MAXPASSSIZE,
                cudaMemcpyHostToDevice) );

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
