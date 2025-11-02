CC=clang
FLAGS=-Wall

all: main.o matrix.o nn.o mnist.o
	$(CC) $(FLAGS) nn.o mnist.o matrix.o main.o -o main -lm

test_mem_leak: test_mem_leak.o nn.o matrix.o mnist.o
	$(CC) $(FLAGS) test_mem_leak.o nn.o matrix.o mnist.o -o test_mem_leak -lm


clean:
	rm *.o main main2 main_str test_mem_leak
