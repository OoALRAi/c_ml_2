CC=clang
FLAGS=-Wall

all: main.o matrix.o nn.o mnist.o
	$(CC) $(FLAGS) nn.o mnist.o matrix.o  main.o -o main -lm

clean:
	rm *.o main main2 main_str test
