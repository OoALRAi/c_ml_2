#include "nn.h"
#include "mnist.h"
#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>

int main()
{

    Dense *d = create_dense(28 * 28, 10, softmax, grad_softmax);
    Loss *loss = create_loss(cross_entropy_loss, grad_cross_entropy_loss);
    Mnist_Dataset *dataset = create_mnist_from_csv("./data/mnist_test.csv");
    int count = 0;
    while (count < 2)
    {
        printf("\n\n=========\ncounter: %d\n=========\n\n", count);
        Mnist_Datapoint *datapoint = mnist_next_datapoint(dataset);
        if (datapoint == NULL)
            break;
        Matrix *gt = new_mat(1, 10);
        Matrix *pred = forward(d, datapoint->data);
        loss_forward(loss, gt, pred);
        loss_backward(loss);
        Matrix *next_grad = loss->grad_error_values;
        backward(d, next_grad, 0.001);
        free_mat(gt);
        count++;
    }
    free_dense(d);
    free_loss(loss);
    free_dataset(dataset);

    return 0;
}