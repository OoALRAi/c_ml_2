#include "nn.h"
#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

void relu(Matrix *input, Matrix *result)
{
    if (input == NULL)
    {
        fprintf(stderr, "input matrix is null\n");
        exit(-1);
    }
    for (size_t y = 0; y < input->rows; y++)
    {
        for (size_t x = 0; x < input->cols; x++)
        {
            double value = get_element_at(input, x, y);
            double act_value = value > 0 ? value : 0;
            set_element_at(result, x, y, act_value);
        }
    }
}

void grad_relu(Matrix *relu_input, Matrix *next_grad, Matrix *result)
{
    for (size_t y = 0; y < relu_input->rows; y++)
    {
        for (size_t x = 0; x < relu_input->cols; x++)
        {
            double value = get_element_at(relu_input, x, y);
            set_element_at(result, x, y, value > 0 ? get_element_at(next_grad, x, y) : 0);
        }
    }
}

double sigmoid_function(double value)
{
    return 1 / (1 + exp(-value));
}

void sigmoid(Matrix *input, Matrix *result)
{
    for (size_t y = 0; y < input->rows; y++)
    {
        for (size_t x = 0; x < input->cols; x++)
        {
            double value = get_element_at(input, x, y);
            set_element_at(result, x, y, sigmoid_function(value));
        }
    }
}

void grad_sigmoid(Matrix *sigmoid_input, Matrix *next_grad, Matrix *result)
{
    for (size_t y = 0; y < sigmoid_input->rows; y++)
    {
        for (size_t x = 0; x < sigmoid_input->cols; x++)
        {
            double value = get_element_at(sigmoid_input, x, y);
            double sig_value = sigmoid_function(value);
            set_element_at(result, x, y, sig_value * (1 - sig_value));
        }
    }
}

void softmax(Matrix *input, Matrix *result)
{
    double max_value = max(input);
    double sum_exp = 0;
    for (size_t y = 0; y < input->rows; y++)
    {
        for (size_t x = 0; x < input->cols; x++)
        {
            double value = get_element_at(input, x, y) - max_value;
            double exp_value = exp(value);
            sum_exp += exp_value;
        }
    }

    for (size_t y = 0; y < input->rows; y++)
    {
        for (size_t x = 0; x < input->cols; x++)
        {
            double value = get_element_at(input, x, y) - max_value;
            double exp_value = exp(value);
            set_element_at(result, x, y, exp_value / sum_exp);
        }
    }
}

void grad_softmax(Matrix *softmax_input, Matrix *ground_truth, Matrix *result)
{
    // the second argument to this function is the ground truth
    // since the gradient of the softmax is computed under the assumption
    // that softmax is use and the cross entropy loss as error function.
    softmax(softmax_input, result);
    sub_mat_to(result, ground_truth, result);
}

void identity_func(Matrix *input, Matrix *result)
{
    copy_mat(input, result);
    printf("###\n%f\n###\n", exp(20));
}
void grad_identity_func(Matrix *input, Matrix *next_grad, Matrix *result)
{
    (void)input;
    copy_mat(next_grad, result);
}

Matrix *mse(Matrix *y, Matrix *y_hat)
{
    Matrix *result = sub_mat(y, y_hat);
    elementwise_pow_mat_to(result, result, 2);
    div_mat_by_value_to(result, result, 2);
    return result;
}
Matrix *grad_mse(Matrix *y, Matrix *y_hat)
{
    Matrix *sub = sub_mat(y, y_hat);
    multiply_mat_with_value_to(sub, sub, -1);
    return sub;
}

Matrix *cross_entropy_loss(Matrix *y, Matrix *y_hat)
{
    // use yi and xi for indexing because y here represents the ground truth.
    // ground truth y is one hot encoded vector of the true label
    // if true label is 2 then it should be represented as:
    // [0,0,1,0]
    double sum = 0;
    for (int yi = 0; yi < y->rows; yi++)
    {
        for (int xi = 0; xi < y->cols; xi++)
        {
            // y_i * log (y_i_hat)
            sum -= get_element_at(y, xi, yi) * log(get_element_at(y_hat, xi, yi));
        }
    }
    Matrix *r = new_mat(1, 1);
    set_element_at(r, 0, 0, sum);
    return r;
}

Matrix *grad_cross_entropy_loss(Matrix *y, Matrix *y_hat)
{
    (void)y_hat;
    return y;
}

Dense *create_dense(int in, int out, void (*activation)(Matrix *, Matrix *), void(grad_activation)(Matrix *, Matrix *, Matrix *))
{
    if (in <= 0 || out <= 0 || activation == NULL && grad_activation == NULL)
    {
        fprintf(stderr, "invalid parameters, faild to create dense layer\n");
        exit(0);
    }
    Dense *dense = calloc(1, sizeof(Dense));
    dense->id = id_counter;
    id_counter++;
    dense->in_dim = in;
    dense->out_dim = out;

    dense->weights = new_mat(in, out);
    dense->bias = new_mat(1, out);
    dense->z = new_mat(1, out);
    dense->output = new_mat(1, out);
    dense->dz = new_mat(1, out);

    random_fill_mat(dense->weights);
    random_fill_mat(dense->bias);

    dense->activation = activation;
    dense->grad_activation = grad_activation;
    return dense;
}

Loss *create_loss(Matrix *(*error_function)(Matrix *, Matrix *), Matrix *(*grad_error_function)(Matrix *, Matrix *))
{
    Loss *loss = malloc(sizeof(Loss));
    loss->error_function = error_function;
    loss->grad_error_function = grad_error_function;
    loss->y = NULL;
    loss->y_hat = NULL;
    loss->error_values = NULL;
    return loss;
}

void print_dense(Dense *d)
{
    printf("dense layer: %d\n", d->id);
    printf("input dim: %d, output dim: %d\n", d->in_dim, d->out_dim);
    if (d->input != NULL)
    {
        printf("input tensor dim: (%dx%d)\n", d->input->rows, d->input->cols);
    }

    printf("bias dim: (%dx%d)\n", d->bias->rows, d->bias->cols);

    if (d->z != NULL)
    {
        printf("z tensor dim: (%dx%d)\n", d->z->rows, d->z->cols);
    }
    if (d->output != NULL)
    {
        printf("output tensor dim: (%dx%d)\n", d->output->rows, d->output->cols);
    }
}

Matrix *forward(Dense *d, Matrix *input)
{
    d->input = input;
    mul_mat_to(input, d->weights, d->z);
    add_mat_to(d->z, d->bias, d->z);

    d->activation(d->z, d->output);

    return d->output;
}

Matrix *backward(Dense *d, Matrix *next_grad, double alpha)
{
    d->grad_activation(d->z, next_grad, d->dz);

    Matrix *dydw = transpose_mat(d->input);
    if (d->dw == NULL)
    {
        Matrix *dw = mul_mat(dydw, d->dz);
        multiply_mat_with_value_to(dw, dw, alpha);
        d->dw = dw;
    }
    else
    {
        mul_mat_to(dydw, d->dz, d->dw);
        multiply_mat_with_value_to(d->dw, d->dw, alpha);
    }
    free_mat(dydw);
    sub_mat_to(d->weights, d->dw, d->weights);

    Matrix *dydx = transpose_mat(d->weights);
    Matrix *dx = mul_mat(next_grad, dydx);
    free_mat(dydx);
    return dx;
}

Matrix *loss_forward(Loss *loss, Matrix *y, Matrix *y_hat)
{
    loss->y = y;
    loss->y_hat = y_hat;
    Matrix *error_values = loss->error_function(y, y_hat);
    loss->error_values = error_values;
    return error_values;
}
Matrix *loss_backward(Loss *loss)
{
    Matrix *grad_error_values = loss->grad_error_function(loss->y, loss->y_hat);
    loss->grad_error_values = grad_error_values;
    return grad_error_values;
}