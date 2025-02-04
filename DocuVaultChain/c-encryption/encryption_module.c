// c-encryption/encryption_module.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KEY 0x5A  // Simple XOR key for demonstration

// Function to encrypt a string
void encrypt(char *data) {
    for (int i = 0; i < strlen(data); i++) {
        data[i] ^= KEY;
    }
}

// Function to decrypt a string (same as encrypt for XOR)
void decrypt(char *data) {
    encrypt(data); // XOR encryption is symmetric
}

int main() {
    char message[256] = "Confidential Document Content";
    printf("Original Message: %s\n", message);

    encrypt(message);
    printf("Encrypted Message: %s\n", message);

    decrypt(message);
    printf("Decrypted Message: %s\n", message);

    return 0;
}
