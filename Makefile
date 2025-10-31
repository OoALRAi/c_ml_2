CC=clang
FLAGS=-Wall

all: main.o matrix.o nn.o mnist.o
	$(CC) $(FLAGS) nn.o mnist.o matrix.o  main.o -o main -lm


main2: mnist.o main2.o matrix.o nn.o
	$(CC) $(FLAGS) matrix.o nn.o mnist.o main2.o -o main2 -lm



clean:
	rm *.o main main2 main_str test
