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
    elementwise_mul_mat_to(next_grad, result, result);
}

double tanh_func(double value)
{
    return tanh(value);
}

void tanh_act(Matrix *input, Matrix *result)
{
    for (size_t y = 0; y < input->rows; y++)
    {
        for (size_t x = 0; x < input->cols; x++)
        {
            double value = get_element_at(input, x, y);
            set_element_at(result, x, y, tanh_func(value));
        }
    }
}

void grad_tanh(Matrix *tanh_input, Matrix *next_grad, Matrix *result)
{
    for (size_t y = 0; y < tanh_input->rows; y++)
    {
        for (size_t x = 0; x < tanh_input->cols; x++)
        {
            double value = get_element_at(tanh_input, x, y);
            value = 1 - powf(tanh_func(value), 2.0);
            value *= get_element_at(next_grad, x, y);
            set_element_at(result, x, y, value);
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
    Matrix *y_cp = new_copy_of(y);
    return y_cp;
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

    dense->input = new_mat(1, in);
    dense->weights = new_mat(in, out);
    dense->bias = new_mat(1, out);
    dense->out_pred_act = new_mat(1, out);
    dense->out_post_act = new_mat(1, out);
    dense->dz = new_mat(1, out);

    random_fill_mat(dense->weights);
    random_fill_mat(dense->bias);

    dense->activation = activation;
    dense->grad_activation = grad_activation;
    return dense;
}
void free_dense(Dense *d)
{
    if (d->input)
    {
        free_mat(d->input);
    }
    if (d->weights)
    {
        free_mat(d->weights);
    }
    if (d->bias)
    {
        free_mat(d->bias);
    }
    if (d->out_pred_act)
    {
        free_mat(d->out_pred_act);
    }
    if (d->out_post_act)
    {
        free_mat(d->out_post_act);
    }
    if (d->dw)
    {
        free_mat(d->dw);
    }
    if (d->db)
    {
        free_mat(d->db);
    }
    if (d->dz)
    {
        free_mat(d->dz);
    }
    if (d->dx)
    {
        free_mat(d->dx);
    }
    free(d);
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

void free_loss(Loss *l)
{
    if (l->error_values)
    {
        free_mat(l->error_values);
    }
    if (l->grad_error_values)
    {
        print_mat(l->grad_error_values);
        free_mat(l->grad_error_values);
    }
    if (l->y != NULL)
    {
        free_mat(l->y);
    }
    if (l->y_hat)
    {
        free_mat(l->y_hat);
    }
    free(l);
}

void print_dense(Dense *d)
{
    printf("dense layer: %d\n", d->id);
    printf("input dim: %d, out_post_act dim: %d\n", d->in_dim, d->out_dim);
    if (d->input != NULL)
    {
        printf("input tensor dim: (%dx%d)\n", d->input->rows, d->input->cols);
    }

    printf("bias dim: (%dx%d)\n", d->bias->rows, d->bias->cols);

    if (d->out_pred_act != NULL)
    {
        printf("out_pred_act tensor dim: (%dx%d)\n", d->out_pred_act->rows, d->out_pred_act->cols);
    }
    if (d->out_post_act != NULL)
    {
        printf("out_post_act tensor dim: (%dx%d)\n", d->out_post_act->rows, d->out_post_act->cols);
    }
}

Matrix *forward(Dense *d, Matrix *input)
{
    copy_mat(input, d->input);
    mul_mat_to(input, d->weights, d->out_pred_act);
    add_mat_to(d->out_pred_act, d->bias, d->out_pred_act);
    d->activation(d->out_pred_act, d->out_post_act);
    return d->out_post_act;
}

void backward(Dense *d, Matrix *next_grad, double lr)
{
    d->grad_activation(d->out_pred_act, next_grad, d->dz);

    // y = xw+b
    Matrix *dydw = transpose_mat(d->input);
    if (d->dw == NULL)
    {
        Matrix *dw = mul_mat(dydw, d->dz);
        multiply_mat_with_value_to(dw, dw, lr);
        d->dw = dw;
    }
    else
    {
        mul_mat_to(dydw, d->dz, d->dw);
        multiply_mat_with_value_to(d->dw, d->dw, lr);
    }
    free_mat(dydw);
    sub_mat_to(d->weights, d->dw, d->weights);

    if (d->db == NULL)
    {
        d->db = new_mat(d->bias->rows, d->bias->cols);
        copy_mat(d->dz, d->db);
        multiply_mat_with_value_to(d->db, d->db, lr);
        sub_mat_to(d->bias, d->db, d->bias);
    }
    else
    {
        copy_mat(d->dz, d->db);
        multiply_mat_with_value_to(d->db, d->db, lr);
        sub_mat_to(d->bias, d->db, d->bias);
    }

    Matrix *dydx = transpose_mat(d->weights);
    Matrix *dx = mul_mat(d->dz, dydx);
    if (d->dx == NULL)
    {
        d->dx = dx;
    }
    else
    {
        copy_mat(dx, d->dx);
        free_mat(dx);
    }

    free_mat(dydx);
}

void loss_forward(Loss *loss, Matrix *y, Matrix *y_hat)
{
    if (loss->y == NULL)
    {
        loss->y = new_mat_like(y);
    }
    copy_mat(y, loss->y);
    if (!loss->y_hat)
    {
        loss->y_hat = new_mat_like(y_hat);
    }
    copy_mat(y_hat, loss->y_hat);
    Matrix *error_values = loss->error_function(y, y_hat);
    if (loss->error_values)
    {
        copy_mat(error_values, loss->error_values);
        free_mat(error_values);
    }
    else
    {
        loss->error_values = error_values;
    }
}

void loss_backward(Loss *loss)
{
    Matrix *grad_error_values = loss->grad_error_function(loss->y, loss->y_hat);
    if (loss->grad_error_values == NULL)
    {
        loss->grad_error_values = grad_error_values;
    }
    else
    {
        copy_mat(grad_error_values, loss->grad_error_values);
        free_mat(grad_error_values);
    }
}