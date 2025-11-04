#ifndef MNIST_H
#define MNIST_H
#include "matrix.h"
#include <stdio.h>

typedef struct Mnist_Datapoint
{
    Matrix *label; // one hot encoded
    Matrix *data;  // [28x28] of value between 0-255
} Mnist_Datapoint;

typedef struct Mnist_Dataset
{
    char *path;
    int size_to_use;
    int current_dp_index;

    Mnist_Datapoint **datapoints;

} Mnist_Dataset;

void one_hot_label(int label, Matrix *result);
int label_from_one_hot(Matrix *one_hot_label);

Mnist_Datapoint *create_datapoint();
Mnist_Dataset *create_mnist_from_csv(char *path, int size_to_use);
char *read_next_line(FILE *fp);
int read_next_dp_from_file(FILE *file, Mnist_Datapoint *dp);

Mnist_Datapoint *get_next_datapoint(Mnist_Dataset *dataset);

void free_datapoint(Mnist_Datapoint *dp);
void free_dataset(Mnist_Dataset *dataset);

#endif