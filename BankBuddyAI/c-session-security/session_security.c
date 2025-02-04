#include <stdio.h>
#include <string.h>

#define SEC_KEY 0x3C  // Simple XOR key for demonstration

// Encrypt/decrypt using XOR (symmetric)
void secure_session(char *sessionToken) {
    for (int i = 0; i < strlen(sessionToken); i++) {
        sessionToken[i] ^= SEC_KEY;
    }
}

int main() {
    char sessionToken[256] = "BankBuddySession123";
    printf("Original Session Token: %s\n", sessionToken);
    
    secure_session(sessionToken);
    printf("Encrypted Session Token: %s\n", sessionToken);
    
    // Decrypt to verify
    secure_session(sessionToken);
    printf("Decrypted Session Token: %s\n", sessionToken);
    
    return 0;
}
