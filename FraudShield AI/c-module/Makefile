# c-module/Makefile
CC = gcc
CFLAGS = -Wall -O2

all: transaction_processor

transaction_processor: transaction_processor.c
	$(CC) $(CFLAGS) -o transaction_processor transaction_processor.c

clean:
	rm -f transaction_processor
