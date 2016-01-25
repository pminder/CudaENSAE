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
    arguments.test = 0;
  	//Thanks to GNU argp :)
  	argp_parse (&argp, argc, argv, 0, 0, &arguments);
  	//Check and get hash size
	const int hashSize = getHashSize(arguments.format);
	if (hashSize == 0) {
		return EXIT_FAILURE;
	}

    //Initialize vector of structs Hash
	vector<Hash> hashes;
    //Else let's load hashes from file
    loadHashes(hashes, arguments.args[0]);
    if (hashes.size() == 0) {
        return EXIT_FAILURE;
    }

	// Convert hashes to char representation in CPU
	char * gpuHashes = NULL;
	gpuHashes = convertHashes(hashes, hashSize);

	//Allocate GPU memory and launch GPU kernels
	char * results = launchKernels(gpuHashes, hashes.size(), arguments.format,
            hashSize, arguments.test);
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
