CC = clang
CFLAGS = -Wno-implicit-function-declaration -std=c11
LDLIBS = -lm

TEST_SLICE_TARGET = test_slice
TEST_SLICE_SRC = test_slice.c matrix.c
TEST_SLICE_OBJ = $(TEST_SLICE_SRC:.c=.o)


$(TEST_SLICE_TARGET) : $(TEST_SLICE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS) 

.PHONY: clean
clean:
	@echo cleaning files..
	rm *.o $(TEST_SLICE_TARGET)
