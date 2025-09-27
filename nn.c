#include "nn.h"
#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>

Matrix *relu(Matrix *input)
{
    if (input == NULL)
    {
        fprintf(stderr, "input matrix is null\n");
        exit(-1);
    }
    Matrix *r = new_mat(input->rows, input->cols);
    for (size_t y = 0; y < input->rows; y++)
    {
        for (size_t x = 0; x < input->cols; x++)
        {
            float value = get_element_at(input, x, y);
            float act_value = value > 0 ? value : 0;
            set_element_at(r, x, y, act_value);
        }
    }
    return r;
}

Matrix *grad_relu(Matrix *relu_input, Matrix *next_grad)
{
    for (size_t y = 0; y < relu_input->rows; y++)
    {
        for (size_t x = 0; x < relu_input->cols; x++)
        {
            float value = get_element_at(relu_input, x, y);
            set_element_at(next_grad, x, y, value > 0 ? get_element_at(next_grad, x, y) : 0);
        }
    }
    return next_grad;
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

Dense *create_dense(int in, int out, Matrix *(*activation)(Matrix *), Matrix *(grad_activation)(Matrix *, Matrix *))
{
    if (in <= 0 || out <= 0 || activation == NULL && grad_activation == NULL)
    {
        fprintf(stderr, "invalid parameters, faild to create dense layer\n");
        exit(0);
    }
    Dense *dense = malloc(sizeof(Dense));
    dense->id = id_counter;
    id_counter++;
    dense->weights = new_mat(in, out);
    dense->bias = new_mat(1, out);

    random_fill_mat(dense->weights);
    random_fill_mat(dense->bias);

    dense->in_dim = in;
    dense->out_dim = out;
    dense->z = NULL;
    dense->output = NULL;

    dense->dw = NULL;
    dense->db = NULL;
    dense->dz = NULL;

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
    Matrix *linear = mul_mat(input, d->weights);
    Matrix *z = add_mat(linear, d->bias);
    d->z = z;
    Matrix *output = d->activation(z);
    d->output = output;
    return z;
}

Matrix *backward(Dense *d, Matrix *next_grad)
{
}

Matrix *loss_forward(Loss *loss, Matrix *y, Matrix *y_hat)
{
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