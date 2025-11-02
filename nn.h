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

Dense *create_dense(int in, int out, void (*activation)(Matrix *, Matrix *), void(grad_activation)(Matrix *, Matrix *, Matrix *));
Loss *create_loss(Matrix *(*error_function)(Matrix *, Matrix *), Matrix *(*grad_error_function)(Matrix *, Matrix *));
void print_dense(Dense *d);
void free_dense(Dense *d);
void free_loss(Loss *l);

void relu(Matrix *input, Matrix *result);
void grad_relu(Matrix *relu_input, Matrix *next_grad, Matrix *result);

double sigmoid_function(double value);
void sigmoid(Matrix *input, Matrix *result);
void grad_sigmoid(Matrix *sigmoid_input, Matrix *next_grad, Matrix *result);

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