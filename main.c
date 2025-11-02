#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "nn.h"
#include "mnist.h"
Matrix *forward_pass(Dense *network[], int num_layers, Mnist_Datapoint *dp)
{
    Matrix *layer_output = dp->data;
    for (size_t layer_index = 0; layer_index < num_layers; layer_index++)
    {
        Dense *d = network[layer_index];
        layer_output = forward(d, layer_output);
    }
    return layer_output;
}

void backward_pass(Dense *network[], int num_layers, Loss *loss, double lr)
{
    loss_backward(loss);

    Matrix *next_grad = loss->grad_error_values;
    for (int layer_index = num_layers - 1; layer_index >= 0; layer_index--)
    {
        Dense *d = network[layer_index];
        backward(d, next_grad, lr);
        next_grad = d->dx;
    }
}

int train(Dense *network[], int num_layers, Loss *loss, Mnist_Datapoint *dp, double lr)
{
    Matrix *pred = forward_pass(network, num_layers, dp);
    int pred_value = argmax(pred);
    int gt_value = argmax(dp->label);
    loss_forward(loss, dp->label, pred);
    backward_pass(network, num_layers, loss, lr);
    return (pred_value == gt_value);
}

int main(void)
{
    int in_dim = 28 * 28;
    int out_dim_1 = 128;
    int out_dim_2 = 32;
    int out_dim = 10;
    int num_layers = 3;
    double learning_rate = 0.0001;

    Dense *network[] = {
        create_dense(in_dim, out_dim_1, tanh_act, grad_tanh),
        create_dense(out_dim_1, out_dim_2, relu, grad_relu),
        create_dense(out_dim_2, out_dim, softmax, grad_softmax)};
    Loss *loss = create_loss(cross_entropy_loss, grad_cross_entropy_loss);

    Mnist_Dataset *dataset = create_mnist_from_csv("./data/mnist_test.csv");
    int correct_predictions = 0;

    for (size_t epoch = 0; epoch < 30; epoch++)
    {
        while (1)
        {
            Mnist_Datapoint *datapoint = mnist_next_datapoint(dataset);
            if (datapoint == NULL)
                break;

            int is_correct = train(network, num_layers, loss, datapoint, learning_rate);
            correct_predictions += is_correct;
        }
        printf("correct predictions: %d\t\n", correct_predictions);
        printf("loss: \t\t%f\n", loss->error_values->data[0]);
        correct_predictions = 0;
    }
    for (int i = 0; i < num_layers; i++)
    {
        Dense *d = network[i];
        free_dense(d);
    }

    free_loss(loss);
    free_dataset(dataset);

    return 0;
}
