#include <stdio.h>
#include <stdlib.h>
#include "cpuUtils.h"

//TODO: if the file does not end with '\n'
int countLines(FILE * p)
{
	int c = 0, count = 0;
	rewind(p);

	do {
		c = getc(p);
		if (c == '\n') {
			count++;
		}
	} while (c != EOF);

	rewind(p);
	return count; 
}

Hash * loadHashes(const char * fileName)
{
	//Open password file
	FILE * passFile = NULL;
	passFile = fopen(fileName, "r");
	if (!passFile) {
		fprintf(stderr, "Could not open file %s\n", fileName);
		return NULL;
	}

	//Get number of lines
	int nLines = countLines(passFile);
	//Allocate aray of structs Hash
	Hash * hashes = (Hash *)calloc(nLines, sizeof(Hash));
	if (!hashes) {
		fprintf(stderr, "Memory Error\n");
		fclose(passFile);
		return NULL;
	}

	//Load password hashes
	for (int i = 0; i < nLines; ++i) {
		fscanf(passFile, FMT_STRING, hashes[i].name, hashes[i].txtHash);
	}

	fclose(passFile);
	return hashes;
}