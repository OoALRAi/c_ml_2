#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "nn.h"
#include "mnist.h"

int main(void)
{
    int in_dim = 28 * 28;
    int out_dim_1 = 50;
    int out_dim_2 = 20;
    int out_dim = 10;
    int num_layers = 3;
    double learning_rate = 0.0001;

    Dense *network[] = {
        create_dense(in_dim, out_dim_1, relu, grad_relu),
        create_dense(out_dim_1, out_dim_2, relu, grad_relu),
        create_dense(out_dim_2, out_dim, softmax, grad_softmax)};
    Loss *loss = create_loss(cross_entropy_loss, grad_cross_entropy_loss);

    Mnist_Dataset *dataset = create_mnist_from_csv("./data/mnist_test.csv");
    int correct_predictions = 0;

    for (size_t epoch = 0; epoch < 10; epoch++)
    {
        while (1)
        {
            Mnist_Datapoint *datapoint = mnist_next_datapoint(dataset);
            if (datapoint == NULL)
                break;
            Matrix *input = datapoint->data;
            Matrix *output = datapoint->label;

            Matrix *layer_output = input;
            for (size_t layer_index = 0; layer_index < num_layers; layer_index++)
            {
                Dense *d = network[layer_index];
                layer_output = forward(d, layer_output);
            }
            int gt = argmax(output);
            int pred = argmax(layer_output);
            if (gt == pred)
                correct_predictions++;

            Matrix *loss_value = loss_forward(loss, output, layer_output);
            Matrix *loss_grad = loss_backward(loss);

            Matrix *next_grad = loss_grad;
            for (int layer_index = num_layers - 1; layer_index >= 0; layer_index--)
            {
                Dense *d = network[layer_index];
                next_grad = backward(d, next_grad, learning_rate);
            }
        }
        printf("correct predictions: %d\n", correct_predictions);
        correct_predictions = 0;
    }

    return 0;
}
