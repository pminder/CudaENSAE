#ifndef H_CPUUTILS
#define H_CPUUTILS 

#include <string>
#include <vector>

//Struct for hash representation
struct Hash
{
    //User id
	std::string name;
    //user hash (an hex encoded C++ string)
	std::string txt;
    //clear text password (this is what we want to find)
	std::string pass;
};

typedef struct Hash Hash;

//Load hashes from file
//Args:
//  - fileName : file containing user id and hashes (separated by ':')
//  - hashes : reference to a vector of hash structs
void loadHashes(std::vector<Hash> & hashes, const char * fileName);

//Get hash size and returns 0 if hash not supported
int getHashSize(const char * hashName);

//Convert all hashes in hashes and returns a pointer to an allocated chunk of
//memory that can be copied on GPU for processing
//Args:
//  - hashes : vector of structs Hash
//  - hashSize : hash size (for memory allocation)
//returns: a pointer to an array of char
char * convertHashes(const std::vector<Hash> & hashes, const int hashSize);

//Convert a hex-encoded C++ string to an array of char (with real values)
//Args:
//  - hash : hex encoded C++ string
//  - hexHash : pointer to an array of char that will be filled with decoded
//    chars
void hash2hex(const std::string & hash, char * hexHash);

//Utility function to display all results in hashes
void displayResults(std::vector<Hash> const& hashes);

//Store results back to vector of structs Hash from an array of chars
void storeResults(const char * results, std::vector<Hash> & hashes);

#endif // H_CPUUTILS
