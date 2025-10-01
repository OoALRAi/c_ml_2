#ifndef NN_H
#define NN_H
#include "matrix.h"

static int id_counter = 0;
typedef struct
{
    int id;
    int in_dim;
    int out_dim;

    Matrix *input;
    Matrix *weights;
    Matrix *bias;
    Matrix *z;      // pre activation tensor
    Matrix *output; // activated output of the dense;

    // gradient tensors
    Matrix *dw;
    Matrix *db;
    Matrix *dz;

    Matrix *(*activation)(Matrix *);
    Matrix *(*grad_activation)(Matrix *, Matrix *);

} Dense;

typedef struct
{
    Matrix *error_values;
    Matrix *grad_error_values;
    Matrix *y;
    Matrix *y_hat;
    Matrix *(*error_function)(Matrix *, Matrix *);
    Matrix *(*grad_error_function)(Matrix *, Matrix *);
} Loss;

Dense *create_dense(int in, int out, Matrix *(*activation)(Matrix *), Matrix *(grad_activation)(Matrix *, Matrix *));
Loss *create_loss(Matrix *(*error_function)(Matrix *, Matrix *), Matrix *(*grad_error_function)(Matrix *, Matrix *));
void print_dense(Dense *d);

Matrix *relu(Matrix *input);
Matrix *grad_relu(Matrix *relu_input, Matrix *next_grad);

Matrix *mse(Matrix *y, Matrix *y_hat);
Matrix *grad_mse(Matrix *y, Matrix *y_hat);

Matrix *forward(Dense *d, Matrix *input);
Matrix *backward(Dense *d, Matrix *next_grad, float alpha);
Matrix *loss_forward(Loss *loss, Matrix *y, Matrix *y_hat);
Matrix *loss_backward(Loss *loss);

#endif