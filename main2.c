#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "mnist.h"

int main(void)
{

    Mnist_Dataset *ds = create_mnist_from_csv("./data/mnist_test.csv");
    Mnist_Datapoint *dp = mnist_next_datapoint(ds);
    while (dp)
    {
        print_mat(dp->label);
        dp = mnist_next_datapoint(ds);
    }

    return 0;
}
