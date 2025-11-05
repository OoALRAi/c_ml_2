CC=clang
FLAGS=-Wall

all: main.o matrix.o nn.o mnist.o statistic_utils.o
	$(CC) $(FLAGS) statistic_utils.o nn.o mnist.o matrix.o main.o -o main -lm

test_conv: nn.o test_conv.o matrix.o
	$(CC) $(FLAGS) test_conv.o nn.o matrix.o -o test_conv -lm

test_slice: test_slice.o matrix.o
	$(CC) $(FLAGS) test_slice.o matrix.o -o test_slice -lm


test_mem_leak: test_mem_leak.o nn.o matrix.o mnist.o
	$(CC) $(FLAGS) test_mem_leak.o nn.o matrix.o mnist.o -o test_mem_leak -lm

test_tanh: test_tanh.o nn.o matrix.o
	$(CC) $(FLAGS) test_tanh.o nn.o matrix.o -o test_tanh -lm

test_mnist: test_mnist.o mnist.o matrix.o
	$(CC) $(FLAGS) test_mnist.o matrix.o mnist.o -o test_mnist -lm

test_statistic_utils: test_statistic_utils.o matrix.o statistic_utils.o
	$(CC) $(FLAGS) test_statistic_utils.o statistic_utils.o matrix.o -o test_statistic_utils -lm
clean:
	rm *.o main main2 main_str test_mem_leak test_statistic_utils test_conv
