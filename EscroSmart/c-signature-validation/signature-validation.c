#include <stdio.h>
#include <string.h>

#define SECRET_KEY 0x4F  // Dummy key for XOR

void validate_signature(char *data, char *signature) {
    char computed[256];
    strcpy(computed, data);
    for (int i = 0; i < strlen(computed); i++) {
        computed[i] ^= SECRET_KEY;
    }
    if (strcmp(computed, signature) == 0) {
        printf("Signature is valid.\n");
    } else {
        printf("Signature is invalid.\n");
    }
}

int main() {
    char data[] = "EscrowData123";
    // For demonstration, compute a dummy signature
    char signature[256];
    strcpy(signature, data);
    for (int i = 0; i < strlen(signature); i++) {
        signature[i] ^= SECRET_KEY;
    }
    
    printf("Data: %s\n", data);
    printf("Computed Signature: %s\n", signature);
    validate_signature(data, signature);
    return 0;
}
