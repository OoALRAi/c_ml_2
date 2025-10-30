#include <stdio.h>
#include "matrix.h"
#include "nn.h"

int main(void)
{
    int in_dim = 1;
    int out_dim_1 = 10;
    int out_dim_2 = 5;
    float learning_rate = 0.0001;

    Dense *network[] = {
        create_dense(in_dim, out_dim_1, relu, grad_relu),
        create_dense(out_dim_1, out_dim_2, relu, grad_relu),
        create_dense(out_dim_2, 1, relu, grad_relu)};
    Loss *loss = create_loss(mse, grad_mse);

    for (size_t epoch = 0; epoch < 1000; epoch++)
    {

        printf("------------\n");
        for (size_t i = 0; i < 10; i++)
        {
            Matrix *input = new_mat(1, 1);
            const_fill_mat(i % 10, input);
            Matrix *output = new_mat(1, 1);
            const_fill_mat((i % 10) * 10, output);

            Matrix *layer_output = input;
            for (size_t layer_index = 0; layer_index < 3; layer_index++)
            {
                Dense *d = network[layer_index];
                layer_output = forward(d, layer_output);
            }

            printf("input: \t %f | \t gt: \t %f | \t pred: \t %f\n", input->data[0], output->data[0], layer_output->data[0]);

            Matrix *loss_value = loss_forward(loss, output, layer_output);
            // printf("loss:\n");
            // print_mat(loss_value);
            printf("loss: %f\n", loss_value->data[0]);
            Matrix *loss_grad = loss_backward(loss);

            Matrix *next_grad = loss_grad;
            for (int layer_index = 2; layer_index >= 0; layer_index--)
            {
                // printf("layer index: %d \n", layer_index);
                Dense *d = network[layer_index];
                next_grad = backward(d, next_grad, learning_rate);
            }
        }
        printf("------------\n");
    }

    return 0;
}
