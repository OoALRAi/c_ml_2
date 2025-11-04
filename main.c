#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "nn.h"
#include "mnist.h"
#include "statistic_utils.h"

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
    loss_forward(loss, dp->label, pred);
    backward_pass(network, num_layers, loss, lr);
    return 0;
}
void test(Dense *network[], int num_layers, Mnist_Datapoint *dp, Confusion_Matrix *confusion_mat)
{
    Matrix *pred = forward_pass(network, num_layers, dp);
    int pred_value = argmax(pred);
    int gt_value = argmax(dp->label);
    add_prediction(confusion_mat, gt_value, pred_value);
}

int main(void)
{
    int in_dim = 28 * 28;
    int out_dim_1 = 128;
    int out_dim = 10;
    int num_layers = 2;
    double learning_rate = 0.001;
    int num_class = 10; // 10 different classes in mnist
    int dataset_size = 5000;
    int trainset_size = (int)(0.7 * dataset_size);

    Dense *network[] = {
        create_dense(in_dim, out_dim_1, leaky_relu, grad_leaky_relu),
        create_dense(out_dim_1, out_dim, softmax, grad_softmax)};
    Loss *loss = create_loss(cross_entropy_loss, grad_cross_entropy_loss);

    Mnist_Dataset *dataset = create_mnist_from_csv("./data/mnist_test.csv", dataset_size);
    double loss_value = 0;
    Confusion_Matrix *confusion_mat = create_confision_matrix(num_class);

    for (size_t epoch = 0; epoch < 200; epoch++)
    {
        while (1)
        {
            Mnist_Datapoint *datapoint = get_next_datapoint(dataset);
            if (datapoint == NULL || dataset->current_dp_index == trainset_size)
                break;

            int is_correct = train(network, num_layers, loss, datapoint, learning_rate);
            loss_value += loss->error_values->data[0];
        }
        printf("===\n[EPOCH %zu]\n", epoch);
        printf("loss: %.4f\n\n", loss_value / trainset_size);
        loss_value = 0;
        while (1)
        {
            Mnist_Datapoint *datapoint = get_next_datapoint(dataset);
            if (datapoint == NULL)
                break;
            test(network, num_layers, datapoint, confusion_mat);
        }
        print_stats(confusion_mat);
        end_epoch(confusion_mat);
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
