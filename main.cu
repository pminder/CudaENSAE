#include <vector>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include "cpuUtils.h"
#include "libgpu.h"

#include "parseArgs.h"

using namespace std;

int main(int argc, char *argv[])
{
	
	//Check command line arguments
	struct arguments arguments;
	//Set argument defaults 
  	arguments.format = "dummy";
  	//Thanks to GNU argp :)
  	argp_parse (&argp, argc, argv, 0, 0, &arguments);
  	//Check and get hash size
	const int hashSize = getHashSize(arguments.format);
	if (hashSize == 0) {
		return EXIT_FAILURE;
	}

    //Initialize cuda events (for measuring performances)
    cudaEvent_t start, stop;
    //Initialize vector of structs Hash
	vector<Hash> hashes;
    //If test speed
    if (arguments.test == 1) {
        //Create test hash corresponding to 10**6 guesses (<03!)
        Hash testHash;
        testHash.name = "speedtest";
        //Put proper hash according to format
        if (strcmp(arguments.format, "md5") == 0) {
            testHash.txt = "7d1f87f31a7ed090af9e492db89cee6f";
        }
        if (strcmp(arguments.format, "dummy") == 0) {
            testHash.txt = "3c303321";
        }
        //Put it in hashes
        hashes.push_back(testHash);
    }
    //Else let's load hashes from file
    else {
        loadHashes(hashes, arguments.args[0]);
        if (hashes.size() == 0) {
            return EXIT_FAILURE;
        }
    }

	// Convert hashes to char representation in CPU
	char * gpuHashes = NULL;
	gpuHashes = convertHashes(hashes, hashSize);

    //If we want to measure performance
    if (arguments.test == 1) {
        HANDLE_ERROR( cudaEventCreate(&start) );
        HANDLE_ERROR( cudaEventCreate(&stop) );
        HANDLE_ERROR( cudaEventRecord(start, 0) );
    }
	
	//Allocate GPU memory and launch GPU kernels
	char * results = launchKernels(gpuHashes, hashes.size(), arguments.format, hashSize);
	if (!results)
	{
		return EXIT_FAILURE;
	}

    //Possibly for measure performance
    if (arguments.test == 1) {
        HANDLE_ERROR( cudaEventRecord(stop, 0) );
        HANDLE_ERROR( cudaEventSynchronize(stop) );

        //Display number of hashes per second
        float elapsedTime(0);
        HANDLE_ERROR( cudaEventElapsedTime(&elapsedTime, start, stop) );
        
        cout << "Performances: " << 1000000.0 / elapsedTime << " hash/ms";
        cout << endl;
        HANDLE_ERROR( cudaEventDestroy(start) );
        HANDLE_ERROR( cudaEventDestroy(stop) );
    }

	//Display results
    storeResults(results, hashes);
	displayResults(hashes);

    //And we are environmentally concerned coders :)
    free(gpuHashes);
	free(results);

	return 0;
}
