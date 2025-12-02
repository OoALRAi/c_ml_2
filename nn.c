#include "nn.h"
#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

void relu(Matrix *input, Matrix *result)
{
    assert(input != NULL);
    assert(result != NULL);
    int ndims = input->ndims;
    int *idx = calloc(ndims, sizeof(int)); // current indices

    for (int count = 0; count < input->size; count++)
    {
        // Compute current offsets using stride and index counter
        int offset_a = 0, offset_c = 0;
        for (int d = 0; d < ndims; d++)
        {
            offset_a += idx[d] * input->stride[d];
            offset_c += idx[d] * result->stride[d];
        }

        double value = input->data[offset_a];
        double act_value = value > 0 ? value : 0;
        result->data[offset_c] = act_value;

        // compute next index
        for (int d = ndims - 1; d >= 0; d--)
        {
            idx[d]++;
            if (idx[d] < input->shape[d])
                break;
            idx[d] = 0;
        }
    }
    free(idx);
}

void grad_relu(Matrix *relu_input, Matrix *next_grad, Matrix *result)
{
    assert(relu_input != NULL);
    assert(result != NULL);
    assert(next_grad != NULL);

    int ndims = relu_input->ndims;
    int *idx = calloc(ndims, sizeof(int)); // current indices

    for (int count = 0; count < relu_input->size; count++)
    {
        // Compute current offsets using stride and index counter
        int offset_a = 0, offset_c = 0, offset_ng = 0;
        for (int d = 0; d < ndims; d++)
        {
            offset_a += idx[d] * relu_input->stride[d];
            offset_c += idx[d] * result->stride[d];
            offset_ng += idx[d] * next_grad->stride[d];
        }

        double value = relu_input->data[offset_a];
        double ng_value = next_grad->data[offset_ng];
        double grad_value = value > 0 ? ng_value : 0;
        result->data[offset_c] = grad_value;

        // compute next index
        for (int d = ndims - 1; d >= 0; d--)
        {
            idx[d]++;
            if (idx[d] < relu_input->shape[d])
                break;
            idx[d] = 0;
        }
    }
    free(idx);
}

void leaky_relu(Matrix *input, Matrix *result)
{
    assert(input != NULL);
    assert(result != NULL);
    int ndims = input->ndims;
    int *idx = calloc(ndims, sizeof(int)); // current indices

    for (int count = 0; count < input->size; count++)
    {
        // Compute current offsets using stride and index counter
        int offset_a = 0, offset_c = 0;
        for (int d = 0; d < ndims; d++)
        {
            offset_a += idx[d] * input->stride[d];
            offset_c += idx[d] * result->stride[d];
        }

        double value = input->data[offset_a];
        double act_value = value > 0 ? value : 0.1 * value;
        result->data[offset_c] = act_value;

        // compute next index
        for (int d = ndims - 1; d >= 0; d--)
        {
            idx[d]++;
            if (idx[d] < input->shape[d])
                break;
            idx[d] = 0;
        }
    }
    free(idx);
}

void grad_leaky_relu(Matrix *input, Matrix *next_grad, Matrix *result)
{
    assert(input != NULL);
    assert(result != NULL);
    assert(next_grad != NULL);

    int ndims = input->ndims;
    int *idx = calloc(ndims, sizeof(int)); // current indices

    for (int count = 0; count < input->size; count++)
    {
        // Compute current offsets using stride and index counter
        int offset_a = 0, offset_c = 0, offset_ng = 0;
        for (int d = 0; d < ndims; d++)
        {
            offset_a += idx[d] * input->stride[d];
            offset_c += idx[d] * result->stride[d];
            offset_ng += idx[d] * next_grad->stride[d];
        }

        double value = input->data[offset_a];
        double ng_value = next_grad->data[offset_ng];
        double grad_value = value > 0 ? ng_value : 0.1 * ng_value;
        result->data[offset_c] = grad_value;

        // compute next index
        for (int d = ndims - 1; d >= 0; d--)
        {
            idx[d]++;
            if (idx[d] < input->shape[d])
                break;
            idx[d] = 0;
        }
    }
    free(idx);
}

double sigmoid_function(double value)
{
    return 1 / (1 + exp(-value));
}

void sigmoid(Matrix *input, Matrix *result)
{
    assert(input != NULL);
    assert(result != NULL);
    int ndims = input->ndims;
    int *idx = calloc(ndims, sizeof(int)); // current indices

    for (int count = 0; count < input->size; count++)
    {
        // Compute current offsets using stride and index counter
        int offset_a = 0, offset_c = 0;
        for (int d = 0; d < ndims; d++)
        {
            offset_a += idx[d] * input->stride[d];
            offset_c += idx[d] * result->stride[d];
        }

        double value = input->data[offset_a];
        double act_value = sigmoid_function(value);
        result->data[offset_c] = act_value;

        // compute next index
        for (int d = ndims - 1; d >= 0; d--)
        {
            idx[d]++;
            if (idx[d] < input->shape[d])
                break;
            idx[d] = 0;
        }
    }
    free(idx);
}

void grad_sigmoid(Matrix *sigmoid_input, Matrix *next_grad, Matrix *result)
{

    assert(sigmoid_input != NULL);
    assert(result != NULL);
    assert(next_grad != NULL);

    int ndims = sigmoid_input->ndims;
    int *idx = calloc(ndims, sizeof(int)); // current indices

    for (int count = 0; count < sigmoid_input->size; count++)
    {
        // Compute current offsets using stride and index counter
        int offset_a = 0, offset_c = 0, offset_ng = 0;
        for (int d = 0; d < ndims; d++)
        {
            offset_a += idx[d] * sigmoid_input->stride[d];
            offset_c += idx[d] * result->stride[d];
            offset_ng += idx[d] * next_grad->stride[d];
        }

        double value = sigmoid_input->data[offset_a];
        double ng_value = next_grad->data[offset_ng];
        double sig_value = sigmoid_function(value);
        double grad_value = sig_value * (1 - sig_value);
        grad_value *= ng_value;

        result->data[offset_c] = grad_value;

        // compute next index
        for (int d = ndims - 1; d >= 0; d--)
        {
            idx[d]++;
            if (idx[d] < sigmoid_input->shape[d])
                break;
            idx[d] = 0;
        }
    }
    free(idx);
}

void softmax(Matrix *input, Matrix *result)
{
    double max_value = max(input);
    double sum_exp = 0;
    // ===
    assert(input != NULL);
    assert(result != NULL);
    int ndims = input->ndims;
    int *idx = calloc(ndims, sizeof(int)); // current indices

    for (int count = 0; count < input->size; count++)
    {
        // Compute current offsets using stride and index counter
        int offset_a = 0;
        for (int d = 0; d < ndims; d++)
        {
            offset_a += idx[d] * input->stride[d];
        }
        double value = input->data[offset_a] - max_value;
        sum_exp += exp(value);

        // compute next index
        for (int d = ndims - 1; d >= 0; d--)
        {
            idx[d]++;
            if (idx[d] < input->shape[d])
                break;
            idx[d] = 0;
        }
    }
    free(idx);

    idx = calloc(ndims, sizeof(int)); // current indices
    for (int count = 0; count < input->size; count++)
    {
        // Compute current offsets using stride and index counter
        int offset_a = 0, offset_r = 0;
        for (int d = 0; d < ndims; d++)
        {
            offset_a += idx[d] * input->stride[d];
            offset_r += idx[d] * result->stride[d];
        }
        double value = input->data[offset_a] - max_value;
        double exp_value = exp(value);
        result->data[offset_r] = exp_value / sum_exp;

        // compute next index
        for (int d = ndims - 1; d >= 0; d--)
        {
            idx[d]++;
            if (idx[d] < input->shape[d])
                break;
            idx[d] = 0;
        }
    }
    free(idx);
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
}
void grad_identity_func(Matrix *input, Matrix *next_grad, Matrix *result)
{
    (void)input;
    copy_mat(next_grad, result);
}

Matrix *mse(Matrix *y, Matrix *y_hat)
{
    Matrix *result = sub_mat(y, y_hat);
    e_pow_mat_to(result, result, 2);
    div_mat_by_value_to(result, 2, result);
    return result;
}
Matrix *grad_mse(Matrix *y, Matrix *y_hat)
{
    Matrix *sub = sub_mat(y, y_hat);
    scale_mat_to(sub, -1, sub);
    return sub;
}

Matrix *cross_entropy_loss(Matrix *y, Matrix *y_hat)
{
    // use yi and xi for indexing because y here represents the ground truth.
    // ground truth y is one hot encoded vector of the true label
    // if true label is 2 then it should be represented as:
    // [0,0,1,0]
    double sum = 0;
    int result_shape[] = {1};
    Matrix *result = new_mat(result_shape, 1);
    int ndims = y->ndims;
    int *idx = calloc(ndims, sizeof(int)); // current indices

    for (int count = 0; count < y->size; count++)
    {
        // Compute current offsets using stride and index counter
        int offset_a = 0, offset_b = 0;
        for (int d = 0; d < ndims; d++)
        {
            offset_a += idx[d] * y->stride[d];
            offset_b += idx[d] * y_hat->stride[d];
        }

        sum -= y->data[offset_a] * log(y_hat->data[offset_b]);

        // compute next index
        for (int d = ndims - 1; d >= 0; d--)
        {
            idx[d]++;
            if (idx[d] < y->shape[d])
                break;
            idx[d] = 0;
        }
    }
    result->data[0] = sum;
    free(idx);
    return result;
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

    int shape_input[2] = {1, in};
    dense->input = new_mat(shape_input, 2);

    int shape_weight[2] = {in, out};
    dense->weights = new_mat(shape_weight, 2);

    int shape_bias[2] = {1, out};
    dense->bias = new_mat(shape_bias, 2);

    int shape_out_pred_act[2] = {1, out};
    dense->out_pred_act = new_mat(shape_out_pred_act, 2);

    int shape_out_post_act[2] = {1, out};
    dense->out_post_act = new_mat(shape_out_post_act, 2);

    int shape_dz[2] = {1, out};
    dense->dz = new_mat(shape_dz, 2);

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

Conv2d *create_conv2d(int input_channels, int output_channels, int kernel_size, int stride, int padding)
{
    Conv2d *conv = malloc(sizeof(Conv2d));
    conv->input_channels = input_channels;
    conv->output_channels = output_channels;
    conv->kernel_size = kernel_size;
    conv->stride = stride;
    conv->padding = padding;

    conv->kernel_weights = malloc(input_channels * output_channels * sizeof(Matrix));
    return conv;
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

void print_conv2d(Conv2d *c)
{
    printf("conv2d: \n");
    printf("input_channels:\t\t%d\n", c->input_channels);
    printf("output_channels:\t%d\n", c->output_channels);
    printf("kernel size:\t\t%d\n", c->kernel_size);
    printf("stride:\t\t\t%d\n", c->stride);
    printf("padding:\t\t%d\n", c->padding);

    printf("\nnum of output images:\t%d\n", c->input_channels * c->output_channels);

    for (int y = 0; y < c->input_channels; y++)
    {
        for (int x = 0; x < c->output_channels; x++)
        {

            printf("channel: %d, kernel: %d\n", y, x);
            print_mat(c->kernel_weights[y * c->output_channels + x]);
        }
    }
}

void free_loss(Loss *l)
{
    if (l->error_values)
    {
        free_mat(l->error_values);
    }
    if (l->grad_error_values)
    {
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
    // printf("dense layer: %d\n", d->id);
    // printf("input dim: %d, out_post_act dim: %d\n", d->in_dim, d->out_dim);
    // if (d->input != NULL)
    // {
    //     printf("input tensor dim: (%dx%d)\n", d->input->rows, d->input->cols);
    // }

    // printf("bias dim: (%dx%d)\n", d->bias->rows, d->bias->cols);

    // if (d->out_pred_act != NULL)
    // {
    //     printf("out_pred_act tensor dim: (%dx%d)\n", d->out_pred_act->rows, d->out_pred_act->cols);
    // }
    // if (d->out_post_act != NULL)
    // {
    //     printf("out_post_act tensor dim: (%dx%d)\n", d->out_post_act->rows, d->out_post_act->cols);
    // }
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
    Matrix *dydw = transpose_mat(d->input, 0, 1);
    if (d->dw == NULL)
    {
        Matrix *dw = mul_mat(dydw, d->dz);
        scale_mat_to(dw, lr, dw);
        d->dw = dw;
    }
    else
    {
        mul_mat_to(dydw, d->dz, d->dw);
        scale_mat_to(d->dw, lr, d->dw);
    }
    free_mat(dydw);
    sub_mat_to(d->weights, d->dw, d->weights);

    if (d->db == NULL)
    {
        d->db = new_mat(d->bias->shape, d->bias->ndims);
        copy_mat(d->dz, d->db);
        scale_mat_to(d->db, lr, d->db);
        sub_mat_to(d->bias, d->db, d->bias);
    }
    else
    {
        copy_mat(d->dz, d->db);
        scale_mat_to(d->db, lr, d->db);
        sub_mat_to(d->bias, d->db, d->bias);
    }

    Matrix *dydx = transpose_mat(d->weights, 0, 1);
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