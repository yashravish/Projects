CC = gcc
CFLAGS = -Wall -Werror -std=c99 -pedantic -D_POSIX_C_SOURCE=200809L
LDFLAGS =

TARGET = mysh
SRCS = mysh.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

test: $(TARGET) myscript.sh
	cat myscript.sh | ./$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all run test clean