#include <vector>
#include <iostream>
#include <stdio.h>

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

	//Load hashes in array of structs Hash
	vector<Hash> hashes;
	loadHashes(hashes, arguments.args[0]);
	if (hashes.size() == 0) {
		return EXIT_FAILURE;
	}

	// Convert hashes to char representation in CPU
	char * gpuHashes = NULL;
	gpuHashes = convertHashes(hashes, hashSize);
	
	//Allocate GPU memory and launch GPU kernels
	char * results = launchKernels(gpuHashes, hashes.size(), arguments.format, hashSize);
	if (!results)
	{
		return EXIT_FAILURE;
	}

	//Display results
    storeResults(results, hashes);
	displayResults(hashes);

    //And we are environmentally concerned coders :)
    free(gpuHashes);
	free(results);

	return 0;
}
