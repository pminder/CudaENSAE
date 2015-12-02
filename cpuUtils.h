#ifndef H_CPUUTILS
#define H_CPUUTILS 

#define USER_MAX_SIZE 20
#define HASH_MAX_SIZE 16 //MD5 = 8, SHA1 = 10, NTLM = 16
#define HASHTXT_MAX_SIZE 32

#define xstr(s) str(s)
#define str(s) #s	
#define FMT_STRING "%" xstr(USER_MAX_SIZE) "[^:]:%" xstr(HASHTXT_MAX_SIZE) "s\n"

struct Hash
{
	char name[USER_MAX_SIZE];
	// char rawHash[HASH_MAX_SIZE];
	char txtHash[HASHTXT_MAX_SIZE];
};

typedef struct Hash Hash;

Hash * loadHashes(const char * fileName);
int countLines(FILE * fp);

#endif // H_CPUUTILS