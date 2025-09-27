#include <stdio.h>
#include "matrix.h"
#include "nn.h"

int main(void)
{
    int in_dim = 28;
    int out_dim = 10;
    Dense *d = create_dense(in_dim, out_dim, relu, grad_relu);
    Matrix *x = new_mat(1, in_dim);
    const_fill_mat(1, x);
    Matrix *out1 = forward(d, x);
    print_dense(d);
    Dense *d2 = create_dense(out_dim, 2, relu, grad_relu);
    forward(d2, out1);
    printf("--------\n");
    print_dense(d2);
    return 0;
}
