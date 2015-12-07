#include <stdlib.h>		
#include <string.h>		
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>

#include "cpuUtils.h"

using namespace std;

int getHashSize(const char * hashName)
{
	if(strcmp(hashName, "dummy") == 0) {
		return 4;
	}

	cerr << "Format not recognized (yet)" << endl;
	return 0;

}

void  loadHashes(vector<Hash> & hashes, const char * fileName)
{
	//Open password file
	ifstream passFile(fileName);
	if (!passFile) {
		cerr << "Could not open file " << fileName << endl;
		return;
	}

	//Load password hashes
	string line;
	while (getline(passFile, line)) {
		size_t sep = line.find(":");
		Hash hash;
		hash.name = line.substr(0,sep);
		hash.txt = line.substr(sep+1);
		hashes.push_back(hash);
	}

	passFile.close();

}

void hash2hex(const string & hash, char * hexHash)
{
	for (int i=0; 2*i < hash.size(); i++)
	{
		string b = hash.substr(2*i,2);
		hexHash[i] = (char)strtoul(b.c_str(),NULL,16);
	}
}

char * convertHashes(const vector <Hash> & hashes, const int hashSize)
{
	int N = hashes.size();
	char * gpuHashes = NULL;
	gpuHashes = (char*)calloc(N*hashSize,sizeof(char));
	if (!gpuHashes) {
		return NULL;
	}
	for (int i=0; i<N; i++) {
		hash2hex(hashes[i].txt,gpuHashes+i*hashSize);
	}
	return gpuHashes;
}