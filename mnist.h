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
    FILE *file;
    // first row is the legend
    Mnist_Datapoint *current_datapoint;

} Mnist_Dataset;

Mnist_Datapoint *create_datapoint();
Mnist_Datapoint *mnist_next_datapoint(Mnist_Dataset *dataset);
Mnist_Dataset *create_mnist_from_csv(char *path);

#endif