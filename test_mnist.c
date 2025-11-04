#include "mnist.h"
#include "matrix.h"

int main()
{
    Mnist_Dataset *dataset = create_mnist_from_csv("./data/mnist_test.csv", 100);
    Mnist_Datapoint *dp = mnist_next_datapoint(dataset);
    print_mat(dp->data);
    dp = mnist_next_datapoint(dataset);
    print_mat(dp->data);
    return 0;
}