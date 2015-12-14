#ifndef H_CPUUTILS
#define H_CPUUTILS 

#include <string>
#include <vector>

struct Hash
{
	std::string name;
	std::string txt;
	std::string pass;
};

typedef struct Hash Hash;

void loadHashes(std::vector<Hash> & hashes, const char * fileName);
int getHashSize(const char * hashName);

char * convertHashes(const std::vector<Hash> & hashes, const int hashSize);
void hash2hex(const std::string & hash, char * hexHash);

void displayResults(const char * results, const int size);

#endif // H_CPUUTILS