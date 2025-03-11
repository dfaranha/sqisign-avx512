CC=gcc
FLAGS = -O2 -march=native -IC
OUTPUT = theta_simd
SRC = main.c fp_lvl3.c
HEADERS = include

all:
	$(CC) $(FLAGS) -o $(OUTPUT) $(SRC) -I $(HEADERS)
	./$(OUTPUT)