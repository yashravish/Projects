// c-module/transaction_processor.c
#include <stdio.h>
#include <time.h>

typedef struct {
    long id;
    double amount;
    time_t timestamp;
} Transaction;

void process_transaction(Transaction txn) {
    // Example heuristic: flag transactions over $10,000
    if (txn.amount > 10000) {
        printf("High-value transaction detected: ID %ld\n", txn.id);
    } else {
        printf("Transaction %ld processed.\n", txn.id);
    }
}

int main() {
    Transaction txn1 = {1001, 15000.00, time(NULL)};
    Transaction txn2 = {1002, 5000.00, time(NULL)};
    process_transaction(txn1);
    process_transaction(txn2);
    return 0;
}
