CC=gcc
FLAGS = -O3 -march=native 
DEBUG = -g -fsanitize=address,undefined

.PHONY: test bench

test:
	$(CC) $(FLAGS) common/*.c test/*.c sqisign/gf/*.c sqisign/ec/*.c sqisign/gluing/*.c sqisign/gf/*.S sqisign/test/*.c sqisign/verif/*.c sqisign/params/*.c sqisign/generic/*.c -o build/test -I include -I test -I sqisign/ec/include -I sqisign/gf/include -I sqisign/gf/test -I sqisign/gluing/include -I sqisign/test -I sqisign/verif -I sqisign/params -I sqisign/generic
	./build/test

bench:
	$(CC) $(FLAGS) -DNDEBUG common/*.c test/KAT.c bench/*.c sqisign/gf/*.c sqisign/ec/*.c sqisign/gluing/*.c sqisign/gf/*.S sqisign/test/*.c sqisign/verif/*.c sqisign/params/*.c sqisign/generic/*.c -o build/bench -I include -I bench -I test -I sqisign/ec/include -I sqisign/gf/include -I sqisign/gf/test -I sqisign/gluing/include -I sqisign/test -I sqisign/verif -I sqisign/params -I sqisign/generic
	./build/bench

sqifp:
	$(CC) $(FLAGS) common/*.c sqisign/gf/*.c  sqisign/gf/*.S sqisign/ec/*.c sqisign/test/*.c sqisign/gf/test/test_fp.c -o build/test_sqi_fp -I include -I sqisign/gf/include -I sqisign/ec/include -I sqisign/test
	./build/test_sqi_fp

sqifp2:
	$(CC) $(FLAGS) common/*.c sqisign/gf/*.c  sqisign/gf/*.S sqisign/ec/*.c sqisign/test/*.c sqisign/gf/test/test_fp2.c -o build/test_sqi_fp -I include -I sqisign/gf/include -I sqisign/ec/include -I sqisign/test
	./build/test_sqi_fp