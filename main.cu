#include <stdio.h>
#include <stdlib.h>

#include "cpuUtils.h"
#include "libgpu.h"

#define USAGE "USAGE: ./main hash.txt"

int main(int argc, char const *argv[])
{
	//Check command line arguments
	if (argc < 2)
	{
		fprintf(stderr, "%s\n", USAGE);
		return EXIT_FAILURE;
	}

	//Load hashes in array of structs Hash
	Hash * hashes = NULL;
	hashes = loadHashes(argv[1]);
	if (!hashes) {
		return EXIT_FAILURE;
	}

	printf("name:%s\n", hashes[1].name);
	printf("hash:%s\n", hashes[1].txtHash);
	
	return 0;
}