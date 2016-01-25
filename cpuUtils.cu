#include <stdlib.h>		
#include <string.h>		
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>

#include "cpuUtils.h"

#define MAXPASSSIZE 9

using namespace std;

int getHashSize(const char * hashName)
{
	if(strcmp(hashName, "dummy") == 0) {
		return 4;
	}

	if(strcmp(hashName, "md5") == 0) {
		return 16;
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

    //Even it is not supposed to be necessary in C++, let's close the file
	passFile.close();

}

void hash2hex(const string & hash, char * hexHash)
{
    //For every couple of characters (1 octet in hex representation)
	for (int i=0; 2*i < hash.size(); i++)
	{
        //Extract hex encoding
		string b = hash.substr(2*i,2);
        //Convert and store
		hexHash[i] = (char)strtoul(b.c_str(),NULL,16);
	}
}

char * convertHashes(const vector <Hash> & hashes, const int hashSize)
{
    //Get constants
	int N = hashes.size();
    //Allocate on cpu an array similar to what will be used on GPU
	char * gpuHashes = NULL;
	gpuHashes = (char*)calloc(N*hashSize,sizeof(char));
	if (!gpuHashes) {
		return NULL;
	}
    //Convert and store each hash
	for (int i=0; i<N; i++) {
		hash2hex(hashes[i].txt,gpuHashes+i*hashSize);
	}

	return gpuHashes;
}

void storeResults(const char * results, vector<Hash> & hashes)
{
    int size = hashes.size();
	for (int i = 0; i < size; ++i)
	{
		hashes[i].pass = results + i*MAXPASSSIZE;
	}
}

void displayResults(vector<Hash> const& hashes)
{
    for (int i = 0; i < hashes.size(); ++i) {
        cout << "[+] " << hashes[i].name << ":" << hashes[i].pass;
        cout << " (" << hashes[i].txt << ")" << endl;
    }
}
