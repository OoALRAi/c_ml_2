CC=clang
FLAGS=-Wall

all: test.o matrix.o nn.o
	$(CC) $(FLAGS) test.o matrix.o nn.o -o test -lm


main2: mnist.o main2.o matrix.o
	$(CC) $(FLAGS) matrix.o mnist.o main2.o -o main2 -lm



clean:
	rm *.o main main2 main_str test
