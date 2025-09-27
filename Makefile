CC=clang
FLAGS=-Wall

all: main.o matrix.o nn.o
	$(CC) $(FLAGS) main.o matrix.o nn.o -o main -lm


clean:
	rm *.o main
