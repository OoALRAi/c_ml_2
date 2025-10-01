#include <stdio.h>
#include "matrix.h"
#include "nn.h"

int main(void)
{
    int in_dim = 1;
    int out_dim_1 = 10;
    Dense *d = create_dense(in_dim, out_dim_1, relu, grad_relu);
    Dense *d2 = create_dense(out_dim_1, 1, relu, grad_relu);
    Loss *loss = create_loss(mse, grad_mse);
    for (size_t epoch; epoch < 500; epoch++)
    {

        for (size_t i = 0; i < 10; i++)
        {
            printf("------------\n");
            Matrix *input = new_mat(1, 1);
            const_fill_mat(i % 10, input);
            Matrix *output = new_mat(1, 1);
            const_fill_mat((i % 10) * 2, output);
            Matrix *out_1 = forward(d, input);
            Matrix *out_2 = forward(d2, out_1);
            print_mat(input);
            print_mat(output);
            print_mat(out_2);

            Matrix *loss_value = loss_forward(loss, output, out_2);
            printf("loss:\n");
            print_mat(loss_value);
            Matrix *next_grad = loss_backward(loss);
            next_grad = backward(d2, next_grad, 0.0001);
            backward(d, next_grad, 0.0001);
            printf("------------\n");
        }
    }

    return 0;
}
