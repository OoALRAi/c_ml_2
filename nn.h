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
    Matrix *out_pred_act; // pre activation tensor
    Matrix *out_post_act; // activated output of the dense;

    // gradient tensors
    Matrix *dw;
    Matrix *db;
    Matrix *dz;
    Matrix *dx; // next_grad for prev layer

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

typedef struct
{
    // no bias use
    int kernel_size;
    int padding;
    int stride;
    int input_channels;
    int output_channels;

    Matrix **input;           // 2D image with number of input_channels
    Matrix **pre_pool_output; // 2D outputs with number of output_channels

    // first assume there are not pooling functions used
    // Matrix *post_pool_output;

    Matrix **kernel_weights; // num of kernels = input_channels x output_channels
    /*
    for rgb image there are 3 input channels
    if output_channels = 5 then for each of the 3 rgb channels there are 5 different kernels
    i.e. 3x5 = 15 kernels
    */

    Matrix *dw;
    Matrix *dx; // next gradient to previous layers

} Conv2d;

Dense *create_dense(int in, int out, void (*activation)(Matrix *, Matrix *), void(grad_activation)(Matrix *, Matrix *, Matrix *));
Conv2d *create_conv2d(int input_channels, int output_channels, int kernel_size, int stride, int padding);
Loss *create_loss(Matrix *(*error_function)(Matrix *, Matrix *), Matrix *(*grad_error_function)(Matrix *, Matrix *));
void print_dense(Dense *d);
void print_conv2d(Conv2d *c);
void free_dense(Dense *d);
void free_conv2d(Conv2d *c);
void free_loss(Loss *l);

void relu(Matrix *input, Matrix *result);
void grad_relu(Matrix *relu_input, Matrix *next_grad, Matrix *result);

double sigmoid_function(double value);
void sigmoid(Matrix *input, Matrix *result);
void grad_sigmoid(Matrix *sigmoid_input, Matrix *next_grad, Matrix *result);

double tanh_func(double value);
void tanh_act(Matrix *input, Matrix *result);
void grad_tanh(Matrix *tanh_input, Matrix *next_grad, Matrix *result);

void leaky_relu(Matrix *input, Matrix *result);
void grad_leaky_relu(Matrix *input, Matrix *next_grad, Matrix *result);

void softmax(Matrix *input, Matrix *result);
void grad_softmax(Matrix *softmax_input, Matrix *ground_truth, Matrix *result);

void identity_func(Matrix *input, Matrix *result);
void grad_identity_func(Matrix *input, Matrix *next_grad, Matrix *result);

Matrix *mse(Matrix *y, Matrix *y_hat);
Matrix *grad_mse(Matrix *y, Matrix *y_hat);

Matrix *cross_entropy_loss(Matrix *y, Matrix *y_hat);
Matrix *grad_cross_entropy_loss(Matrix *y, Matrix *y_hat);

Matrix *forward(Dense *d, Matrix *input);
void backward(Dense *d, Matrix *next_grad, double lr);
void loss_forward(Loss *loss, Matrix *y, Matrix *y_hat);
void loss_backward(Loss *loss);

#endif