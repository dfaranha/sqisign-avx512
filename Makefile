CC=gcc
FLAGS = -O2 -march=native -IC
OUTPUT = theta_simd
SRC = *.c test/*.c

all:
	$(CC) $(FLAGS) -o $(OUTPUT) $(SRC) -I include -I test
	./$(OUTPUT)