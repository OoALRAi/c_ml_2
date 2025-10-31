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
    double learning_rate = 0.001;

    Dense *network[] = {
        // create_dense(28 * 28, 128, relu, grad_relu),
        create_dense(28 * 28, 10, softmax, grad_softmax)};
    Loss *loss = create_loss(cross_entropy_loss, grad_cross_entropy_loss);

    Mnist_Dataset *dataset = create_mnist_from_csv("./data/mnist_test.csv");
    int correct_predictions = 0;

    for (size_t epoch = 0; epoch < 20; epoch++)
    {

        while (1)
        {

            Mnist_Datapoint *datapoint = mnist_next_datapoint(dataset);
            if (datapoint == NULL)
                break;
            Matrix *input = datapoint->data;
            Matrix *output = datapoint->label;

            Matrix *layer_output = input;
            for (size_t layer_index = 0; layer_index < 1; layer_index++)
            {
                Dense *d = network[layer_index];
                layer_output = forward(d, layer_output);
            }
            int gt = max_arg(output);
            int pred = max_arg(layer_output);
            if (gt == pred)
                correct_predictions++;
            // printf("pred: %d , gt: %d\n", pred, gt);
            // print_mat(layer_output);

            Matrix *loss_value = loss_forward(loss, output, layer_output);
            // printf("loss: %f\n", loss_value->data[0]);
            Matrix *loss_grad = loss_backward(loss);

            Matrix *next_grad = loss_grad;
            for (int layer_index = 0; layer_index >= 0; layer_index--)
            {
                // printf("layer index: %d \n", layer_index);
                Dense *d = network[layer_index];
                next_grad = backward(d, next_grad, learning_rate);
            }
        }
        printf("correct predictions: %d\n", correct_predictions);
        correct_predictions = 0;
    }

    return 0;
}
