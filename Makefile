CC = gcc
CFLAGS = -O3 -march=native -flto=auto #-Wall -Wextra
TEST_CFLAGS = $(CFLAGS) -fsanitize=address,undefined -fno-omit-frame-pointer -g
BENCH_CFLAGS = $(CFLAGS) -DNDEBUG

# ---------------------------------------------------------
# Source files
# ---------------------------------------------------------

CORE_SRCS  = $(wildcard common/*.c shake_avx/*.c sqisign/gf/lvlx/*.c sqisign/ec/*.c sqisign/gluing/*.c sqisign/test/*.c sqisign/verif/*.c sqisign/generic/*.c gen_fp2/*.c)

LVL1_C_SRCS = $(wildcard sqisign/gf/lvl1/*.c sqisign/params/lvl1/*.c)
LVL1_S_SRCS = $(wildcard sqisign/gf/lvl1/*.S)

LVL3_C_SRCS = $(wildcard sqisign/gf/lvl3/*.c sqisign/params/lvl3/*.c)
LVL3_S_SRCS = $(wildcard sqisign/gf/lvl3/*.S)

LVL5_C_SRCS = $(wildcard sqisign/gf/lvl5/*.c sqisign/params/lvl5/*.c) 
LVL5_S_SRCS = $(wildcard sqisign/gf/lvl5/*.S)

TEST_SRCS  = $(wildcard test/*.c)
BENCH_SRCS = $(wildcard bench/*.c)

# ---------------------------------------------------------
# headers
# ---------------------------------------------------------

CORE_I  = -I include -I shake_avx -I sqisign/gf/lvlx -I sqisign/ec/include -I sqisign/gluing/include -I sqisign/test -I sqisign/verif -I sqisign/generic -I gen_fp2
LVL1_I  = -I sqisign/gf/lvl1 -I sqisign/params/lvl1
LVL3_I  = -I sqisign/gf/lvl3 -I sqisign/params/lvl3
LVL5_I  = -I sqisign/gf/lvl5 -I sqisign/params/lvl5
TEST_I  = -I test
BENCH_I = -I bench

SHARED_I = $(TEST_I) $(BENCH_I)

# ---------------------------------------------------------
# object files
# ---------------------------------------------------------
TEST_CORE_OBJS_LVL1 = $(patsubst %.c, build/test/lvl1/%.o, $(CORE_SRCS))
TEST_LVL1_C_OBJS    = $(patsubst %.c, build/test/lvl1/%.o, $(LVL1_C_SRCS))
TEST_LVL1_S_OBJS    = $(patsubst %.S, build/test/lvl1/%.o, $(LVL1_S_SRCS))
TEST_OBJS_LVL1      = $(patsubst %.c, build/test/lvl1/%.o, $(TEST_SRCS))

TEST_CORE_OBJS_LVL3 = $(patsubst %.c, build/test/lvl3/%.o, $(CORE_SRCS))
TEST_LVL3_C_OBJS    = $(patsubst %.c, build/test/lvl3/%.o, $(LVL3_C_SRCS))
TEST_LVL3_S_OBJS    = $(patsubst %.S, build/test/lvl3/%.o, $(LVL3_S_SRCS))
TEST_OBJS_LVL3      = $(patsubst %.c, build/test/lvl3/%.o, $(TEST_SRCS))

TEST_CORE_OBJS_LVL5 = $(patsubst %.c, build/test/lvl5/%.o, $(CORE_SRCS))
TEST_LVL5_C_OBJS    = $(patsubst %.c, build/test/lvl5/%.o, $(LVL5_C_SRCS))
TEST_LVL5_S_OBJS    = $(patsubst %.S, build/test/lvl5/%.o, $(LVL5_S_SRCS))
TEST_OBJS_LVL5      = $(patsubst %.c, build/test/lvl5/%.o, $(TEST_SRCS))

BENCH_CORE_OBJS_LVL1 = $(patsubst %.c, build/bench/lvl1/%.o, $(CORE_SRCS))
BENCH_LVL1_C_OBJS    = $(patsubst %.c, build/bench/lvl1/%.o, $(LVL1_C_SRCS))
BENCH_LVL1_S_OBJS    = $(patsubst %.S, build/bench/lvl1/%.o, $(LVL1_S_SRCS))
BENCH_OBJS_LVL1      = $(patsubst %.c, build/bench/lvl1/%.o, $(BENCH_SRCS))

BENCH_CORE_OBJS_LVL3 = $(patsubst %.c, build/bench/lvl3/%.o, $(CORE_SRCS))
BENCH_LVL3_C_OBJS    = $(patsubst %.c, build/bench/lvl3/%.o, $(LVL3_C_SRCS))
BENCH_LVL3_S_OBJS    = $(patsubst %.S, build/bench/lvl3/%.o, $(LVL3_S_SRCS))
BENCH_OBJS_LVL3      = $(patsubst %.c, build/bench/lvl3/%.o, $(BENCH_SRCS))

BENCH_CORE_OBJS_LVL5 = $(patsubst %.c, build/bench/lvl5/%.o, $(CORE_SRCS))
BENCH_LVL5_C_OBJS    = $(patsubst %.c, build/bench/lvl5/%.o, $(LVL5_C_SRCS))
BENCH_LVL5_S_OBJS    = $(patsubst %.S, build/bench/lvl5/%.o, $(LVL5_S_SRCS))
BENCH_OBJS_LVL5      = $(patsubst %.c, build/bench/lvl5/%.o, $(BENCH_SRCS))

# ---------------------------------------------------------
# commands
# ---------------------------------------------------------
.PHONY: all clean lvl1 lvl3 lvl5

# RECURSIVE MAKE: Forces sequential building to prevent Python code generation race conditions
all:
	$(MAKE) lvl1
	$(MAKE) lvl3
	$(MAKE) lvl5

lvl1: build/test_lvl1 build/bench_lvl1
lvl3: build/test_lvl3 build/bench_lvl3
lvl5: build/test_lvl5 build/bench_lvl5

# ---------------------------------------------------------
# CODE GENERATION STATE TRACKING
# ---------------------------------------------------------
# These hidden files act as switches. If you run `make lvl3`, it deletes the lvl1 file
# and creates the lvl3 file, ensuring the Python script only runs when the level changes.
build/.state_lvl1:
	@echo ">>> Generating lvl1 code..."
	@mkdir -p $(@D)
	python3 gen_fp2/gen.py lvl1
	@rm -f build/.state_lvl*
	@touch $@

build/.state_lvl3:
	@echo ">>> Generating lvl3 code..."
	@mkdir -p $(@D)
	python3 gen_fp2/gen.py lvl3
	@rm -f build/.state_lvl*
	@touch $@

build/.state_lvl5:
	@echo ">>> Generating lvl5 code..."
	@mkdir -p $(@D)
	python3 gen_fp2/gen.py lvl5
	@rm -f build/.state_lvl*
	@touch $@

# ---------------------------------------------------------
# Executable Linking (Unchanged)
# ---------------------------------------------------------
build/test_lvl1: $(TEST_CORE_OBJS_LVL1) $(TEST_LVL1_C_OBJS) $(TEST_LVL1_S_OBJS) $(TEST_OBJS_LVL1)
	@mkdir -p $(@D)
	$(CC) $(TEST_CFLAGS) $^ -o $@

build/test_lvl3: $(TEST_CORE_OBJS_LVL3) $(TEST_LVL3_C_OBJS) $(TEST_LVL3_S_OBJS) $(TEST_OBJS_LVL3)
	@mkdir -p $(@D)
	$(CC) $(TEST_CFLAGS) $^ -o $@

build/test_lvl5: $(TEST_CORE_OBJS_LVL5) $(TEST_LVL5_C_OBJS) $(TEST_LVL5_S_OBJS) $(TEST_OBJS_LVL5)
	@mkdir -p $(@D)
	$(CC) $(TEST_CFLAGS) $^ -o $@

build/bench_lvl1: $(BENCH_CORE_OBJS_LVL1) $(BENCH_LVL1_C_OBJS) $(BENCH_LVL1_S_OBJS) $(BENCH_OBJS_LVL1)
	@mkdir -p $(@D)
	$(CC) $(BENCH_CFLAGS) $^ -o $@

build/bench_lvl3: $(BENCH_CORE_OBJS_LVL3) $(BENCH_LVL3_C_OBJS) $(BENCH_LVL3_S_OBJS) $(BENCH_OBJS_LVL3)
	@mkdir -p $(@D)
	$(CC) $(BENCH_CFLAGS) $^ -o $@

build/bench_lvl5: $(BENCH_CORE_OBJS_LVL5) $(BENCH_LVL5_C_OBJS) $(BENCH_LVL5_S_OBJS) $(BENCH_OBJS_LVL5)
	@mkdir -p $(@D)
	$(CC) $(BENCH_CFLAGS) $^ -o $@

clean:
	rm -rf build

# ---------------------------------------------------------
# pattern rules (TEST)
# ---------------------------------------------------------
# Notice the `| build/.state_lvl1` at the end of the first line.
# This means "Make sure build/.state_lvl1 exists before proceeding."
build/test/lvl1/%.o: %.c | build/.state_lvl1
	@mkdir -p $(@D)
	$(CC) $(TEST_CFLAGS) $(CORE_I) $(LVL1_I) $(SHARED_I) -c $< -o $@

build/test/lvl1/%.o: %.S | build/.state_lvl1
	@mkdir -p $(@D)
	$(CC) $(TEST_CFLAGS) $(CORE_I) $(LVL1_I) $(SHARED_I) -c $< -o $@

build/test/lvl3/%.o: %.c | build/.state_lvl3
	@mkdir -p $(@D)
	$(CC) $(TEST_CFLAGS) $(CORE_I) $(LVL3_I) $(SHARED_I) -c $< -o $@

build/test/lvl3/%.o: %.S | build/.state_lvl3
	@mkdir -p $(@D)
	$(CC) $(TEST_CFLAGS) $(CORE_I) $(LVL3_I) $(SHARED_I) -c $< -o $@

build/test/lvl5/%.o: %.c | build/.state_lvl5
	@mkdir -p $(@D)
	$(CC) $(TEST_CFLAGS) $(CORE_I) $(LVL5_I) $(SHARED_I) -c $< -o $@

build/test/lvl5/%.o: %.S | build/.state_lvl5
	@mkdir -p $(@D)
	$(CC) $(TEST_CFLAGS) $(CORE_I) $(LVL5_I) $(SHARED_I) -c $< -o $@

# ---------------------------------------------------------
# pattern rules (BENCH)
# ---------------------------------------------------------
build/bench/lvl1/%.o: %.c | build/.state_lvl1
	@mkdir -p $(@D)
	$(CC) $(BENCH_CFLAGS) $(CORE_I) $(LVL1_I) $(SHARED_I) -c $< -o $@

build/bench/lvl1/%.o: %.S | build/.state_lvl1
	@mkdir -p $(@D)
	$(CC) $(BENCH_CFLAGS) $(CORE_I) $(LVL1_I) $(SHARED_I) -c $< -o $@

build/bench/lvl3/%.o: %.c | build/.state_lvl3
	@mkdir -p $(@D)
	$(CC) $(BENCH_CFLAGS) $(CORE_I) $(LVL3_I) $(SHARED_I) -c $< -o $@

build/bench/lvl3/%.o: %.S | build/.state_lvl3
	@mkdir -p $(@D)
	$(CC) $(BENCH_CFLAGS) $(CORE_I) $(LVL3_I) $(SHARED_I) -c $< -o $@

build/bench/lvl5/%.o: %.c | build/.state_lvl5
	@mkdir -p $(@D)
	$(CC) $(BENCH_CFLAGS) $(CORE_I) $(LVL5_I) $(SHARED_I) -c $< -o $@

build/bench/lvl5/%.o: %.S | build/.state_lvl5
	@mkdir -p $(@D)
	$(CC) $(BENCH_CFLAGS) $(CORE_I) $(LVL5_I) $(SHARED_I) -c $< -o $@
