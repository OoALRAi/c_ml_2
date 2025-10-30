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

    void (*activation)(Matrix *, Matrix *);
    void (*grad_activation)(Matrix *, Matrix *, Matrix *);

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

Dense *create_dense(int in, int out, void (*activation)(Matrix *, Matrix *), void(grad_activation)(Matrix *, Matrix *, Matrix *));
Loss *create_loss(Matrix *(*error_function)(Matrix *, Matrix *), Matrix *(*grad_error_function)(Matrix *, Matrix *));
void print_dense(Dense *d);

void relu(Matrix *input, Matrix *result);
void grad_relu(Matrix *relu_input, Matrix *next_grad, Matrix *result);

float sigmoid_function(float value);
void sigmoid(Matrix *input, Matrix *result);
void grad_sigmoid(Matrix *sigmoid_input, Matrix *next_grad, Matrix *result);

void softmax(Matrix *input, Matrix *result);
void grad_softmax(Matrix *softmax_input, Matrix *ground_truth, Matrix *result);

Matrix *mse(Matrix *y, Matrix *y_hat);
Matrix *grad_mse(Matrix *y, Matrix *y_hat);

Matrix *cross_entropy_loss(Matrix *y, Matrix *y_hat);
Matrix *grad_cross_entropy_loss(Matrix *y, Matrix *y_hat);

Matrix *forward(Dense *d, Matrix *input);
Matrix *backward(Dense *d, Matrix *next_grad, float alpha);
Matrix *loss_forward(Loss *loss, Matrix *y, Matrix *y_hat);
Matrix *loss_backward(Loss *loss);

#endif