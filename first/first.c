#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>


#define MAX_LINE 100


typedef struct {
    unsigned long long tag;
    int VALID;
    int thelastUsedOne;
    int insertionInOrder;
} CacheLine;


typedef struct {
    CacheLine *lines;
    int numberOFSets;
    int itsAssociativity;
    int BlockSize;
    int indexBits;
    int offsetBits;
} Cache;


Cache* initializeCache(int cacheSize, int associativity, int blockSize);
int accessCache(Cache *cache, unsigned long long address, char *replacementPolicy);
void updateCache(Cache *cache, unsigned long long address, char *replacementPolicy);
void simulateCache(Cache *cache, char *traceFile, char *replacementPolicy);




int main(int argc, char *argv[]) {
    int five = 5;
    if (argc != five) {
        fprintf(stderr, "Usage: %s <cachesize> <assoc:n> <block size> <trace file>\n", argv[0]);
        return 1;
    }


    int one = 1;
    int two = 2;
    int theCacheSize = atoi(argv[one]);
    char *ASSOCSTR = strchr(argv[two], ':');


    if (ASSOCSTR == NULL) {
    fprintf(stderr, "This is an invalid associativity format\n");
    return EXIT_FAILURE;
}


    int theAssociativity = atoi(ASSOCSTR + 1);
    int blockSize = atoi(argv[3]);
    char *traceFile = argv[4];


    if (theCacheSize <= 0 || theAssociativity <= 0 || blockSize <= 0) {
        fprintf(stderr, "This is an invalid cache parameters\n");
        return 1;
    }


    Cache *cache = initializeCache(theCacheSize, theAssociativity, blockSize);


    printf("\nLRU\n");
    simulateCache(cache, traceFile, "LRU");


    free(cache->lines);
    free(cache);
    cache = initializeCache(theCacheSize, theAssociativity, blockSize);


    printf("\nFIFO\n");
    simulateCache(cache, traceFile, "FIFO");


    free(cache->lines);
    free(cache);
    return 0;
}


Cache* initializeCache(int cacheSize, int associativity, int blockSize) {
    Cache *cache = (Cache*)malloc(sizeof(Cache));
    if (!cache) {
        fprintf(stderr, "Memory allocation failed for cache\n");
        exit(1);
    }


    cache->itsAssociativity = associativity;
    cache->BlockSize = blockSize;
    cache->numberOFSets = cacheSize / (associativity * blockSize);
    cache->offsetBits = log2(blockSize);
    cache->indexBits = log2(cache->numberOFSets);


    cache->lines = (CacheLine*)calloc(cache->numberOFSets * associativity, sizeof(CacheLine));
    if (!cache->lines) {
        fprintf(stderr, "Memory allocation failed for cache lines\n");
        free(cache);
        exit(1);
    }


    return cache;
}


int accessCache(Cache *cache, unsigned long long address, char *replacementPolicy) {
    unsigned long long tag = address >> (cache->indexBits + cache->offsetBits);
    int index = (address >> cache->offsetBits) & ((1 << cache->indexBits) - 1);
    int setStart = index * cache->itsAssociativity;


    int I = 0;
    while (I < cache->itsAssociativity) {
        if (cache->lines[setStart + I].VALID && cache->lines[setStart + I].tag == tag) {
            if (strcmp(replacementPolicy, "LRU") == 0) {
                cache->lines[setStart + I].thelastUsedOne = 0;
                int j = 0;
                while (j < cache->itsAssociativity) {
                    if (j != I && cache->lines[setStart + j].VALID) {
                        cache->lines[setStart + j].thelastUsedOne++;
                    }
                    j++;
                }
            }
            return 1;
        }
        I++;
    }
    return 0;
}




void updateCache(Cache *cache, unsigned long long address, char *replacementPolicy) {
    unsigned long long tag = address >> (cache->indexBits + cache->offsetBits);
    int index = (address >> cache->offsetBits) & ((1 << cache->indexBits) - 1);
    int setStart = index * cache->itsAssociativity;
    int replacementIndex = -1;


    int i = 0;
    while (i < cache->itsAssociativity) {
        if (!cache->lines[setStart + i].VALID) {
            replacementIndex = i;
            break;
        }
        i++;
    }


    if (replacementIndex == -1) {
        if (strcmp(replacementPolicy, "LRU") == 0) {
            int maxLastUsed = -1;
            int j = 0;
            while (j < cache->itsAssociativity) {
                if (cache->lines[setStart + j].thelastUsedOne > maxLastUsed) {
                    maxLastUsed = cache->lines[setStart + j].thelastUsedOne;
                    replacementIndex = j;
                }
                j++;
            }
        } else {
            int maxInsertionOrder = -1;
            int k = 0;
            while (k < cache->itsAssociativity) {
                if (cache->lines[setStart + k].insertionInOrder > maxInsertionOrder) {
                    maxInsertionOrder = cache->lines[setStart + k].insertionInOrder;
                    replacementIndex = k;
                }
                k++;
            }
        }
    }


    cache->lines[setStart + replacementIndex].tag = tag;
    cache->lines[setStart + replacementIndex].VALID = 1;
   
    if (strcmp(replacementPolicy, "LRU") == 0) {
        cache->lines[setStart + replacementIndex].thelastUsedOne = 0;
        int l = 0;
        while (l < cache->itsAssociativity) {
            if (l != replacementIndex && cache->lines[setStart + l].VALID) {
                cache->lines[setStart + l].thelastUsedOne++;
            }
            l++;
        }
    } else {
        static int globalInsertionOrder = 0;
        cache->lines[setStart + replacementIndex].insertionInOrder = globalInsertionOrder++;
    }
}




void simulateCache(Cache *cache, char *traceFile, char *replacementPolicy) {
    FILE *thefile = fopen(traceFile, "r");
    if (!thefile) {
        fprintf(stderr, "Unable to open trace file: %s\n", traceFile);
        exit(1);
    }


    char operation;
    unsigned long long address;
    int totalAccesses = 0;
    int hits = 0;


    while (fscanf(thefile, " %c %llx", &operation, &address) == 2) {
        totalAccesses++;
        if (accessCache(cache, address, replacementPolicy)) {
            hits++;
        } else {
            updateCache(cache, address, replacementPolicy);
        }
    }


    fclose(thefile);


    int totalmisses = totalAccesses - hits;
    double hitRatio = (double)hits / totalAccesses * 100;
    double missRatio = 100.0 - hitRatio;


    printf("Total Hits: %d\n", hits);
    printf("Total Misses: %d\n", totalmisses);
    printf("Hit Ratio: %.2f%%\n", hitRatio);
    printf("Miss Ratio: %.2f%%\n", missRatio);
}