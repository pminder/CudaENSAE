#include <vector>
#include <iostream>

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

	cout << hashes[0].name << endl;
	cout << hashes[0].txt << endl;
	cout << hashes[1].name << endl;
	cout << hashes[1].txt << endl;

	char * gpuHashes = NULL;
	gpuHashes = convertHashes(hashes, hashSize);
	for (int i = 0; i < 8; ++i)
	{
		cout << *(gpuHashes+i) <<" ";
	}
	return 0;
}