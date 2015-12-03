#include <stdio.h>
#include <stdlib.h>

#include "cpuUtils.h"
#include "libgpu.h"

#include "parseArgs.h"

int main(int argc, char *argv[])
{
	//Check command line arguments
	struct arguments arguments;
	//Set argument defaults 
  	arguments.format = "dummy";
  	//Thanks to GNU argp :)
  	argp_parse (&argp, argc, argv, 0, 0, &arguments);

	//Load hashes in array of structs Hash
	Hash * hashes = NULL;
	hashes = loadHashes(arguments.args[0]);
	if (!hashes) {
		return EXIT_FAILURE;
	}

	printf("name:%s\n", hashes[0].name);
	printf("hash:%s\n", hashes[0].txtHash);
	printf("name:%s\n", hashes[1].name);
	printf("hash:%s\n", hashes[1].txtHash);
	printf("name:%s\n", hashes[2].name);
	printf("hash:%s\n", hashes[2].txtHash);
	
	return 0;
}