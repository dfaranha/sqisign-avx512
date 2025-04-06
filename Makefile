CC=gcc
FLAGS = -O2 -march=native

.PHONY: test bench

test:
	$(CC) $(FLAGS) common/*.c test/*.c -o build/test -I include -I test
	./build/test

bench:
	$(CC) $(FLAGS) common/*.c bench/*.c -o build/bench -I include -I bench
	./build/bench
