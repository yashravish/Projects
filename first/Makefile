CC = gcc

CFLAGS = -Wall -Werror -fsanitize=address,undefined -g -lm

TARGET = first.out

SRC = first.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)