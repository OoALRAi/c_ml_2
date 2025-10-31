#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "mnist.h"
#include "nn.h"

int main(void)
{

    Mnist_Dataset *ds = create_mnist_from_csv("./data/mnist_test.csv");
    Mnist_Datapoint *dp;
    int count = 0;
    while (1)
    {
        dp = mnist_next_datapoint(ds);
        if (dp == NULL)
            break;
        count++;
        int label = label_from_one_hot(dp->label);
        printf("label: %d\n", label);
    }

    printf("count: %d\n", count);
    // Dense *dense = create_dense(28 * 28, 10, softmax, grad_softmax);
    // Loss *loss = create_loss(cross_entropy_loss, grad_cross_entropy_loss);

    // Matrix *output = forward(dense, dp->data);
    // print_mat(output);

    return 0;
}
