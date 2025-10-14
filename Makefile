CC=gcc
FLAGS = -O3 -march=native

.PHONY: test bench

test:
	$(CC) $(FLAGS) common/*.c test/*.c sqisign-gf/*.c sqisign-gf/*.S sqisign-gf/test/test_utils.c -o build/test -I include -I test -I sqisign-gf/include -I sqisign-gf/test
	./build/test

bench:
	$(CC) $(FLAGS) common/*.c bench/*.c sqisign-gf/*.c sqisign-gf/*.S sqisign-gf/test/test_utils.c -o build/bench -I include -I bench -I sqisign-gf/include -I sqisign-gf/test
	./build/bench

sqifp:
	$(CC) $(FLAGS) common/*.c sqisign-gf/*.c  sqisign-gf/*.S sqisign-gf/test/test_utils.c sqisign-gf/test/test_fp.c -o build/test_sqi_fp -I include -I sqisign-gf/include -I sqisign-gf/test
	./build/test_sqi_fp

sqifp2:
	$(CC) $(FLAGS) common/*.c sqisign-gf/*.c  sqisign-gf/*.S sqisign-gf/test/test_utils.c sqisign-gf/test/test_fp2.c -o build/test_sqi_fp -I include -I sqisign-gf/include -I sqisign-gf/test
	./build/test_sqi_fp