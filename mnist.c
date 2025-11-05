#include "mnist.h"
#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

char *skip_value(char *str)
{
    char c;
    while (1)
    {
        c = str[0];
        if (c == '\n')
            return str;
        if (c == '\0')
        {
            return NULL;
        }
        if (c == ',')
        {
            str++;
            return str;
        }
        str++;
    }
}
char *read_next_line(FILE *fp)
{
    char *line = NULL;
    size_t len = 0; // buffer size
    ssize_t read;   // number of chars
    read = getline(&line, &len, fp);
    if (read == -1)
    {
        free(line);
        return NULL;
    }
    return line;
}

void one_hot_label(int label, Matrix *result)
{
    for (int i = 0; i < result->cols; i++)
    {
        SET_ELEMENT_AT(result, i, 0, i == label);
    }
}
int label_from_one_hot(Matrix *one_hot_label)
{
    int label = -1;
    for (int i = 0; i < one_hot_label->cols; i++)
    {
        if (GET_ELEMENT_AT(one_hot_label, i, 0) == 1)
        {
            label = i;
        }
    }
    return label;
}

void parse_line_to_mat(char *line_data, Matrix *data, Matrix *label)
{
    if (!line_data)
    {
        fprintf(stderr, "line data is null!\n");
        exit(0);
    }
    if (!data)
    {
        fprintf(stderr, "data matrix is null!\n");
        exit(0);
    }
    if (!label)
    {
        fprintf(stderr, "label matrix is null!\n");
        exit(0);
    }

    char *current_char = line_data;
    double label_value = atof(current_char);
    one_hot_label(label_value, label);
    current_char = skip_value(current_char);
    for (int i = 0; i < 28 * 28; i++)
    {
        double value = atof(current_char);
        value /= 256;

        SET_ELEMENT_AT(data, i, 0, value);
        current_char = skip_value(current_char);
        if (current_char == NULL)
            break;
    }
}

Mnist_Datapoint *create_datapoint()
{
    Mnist_Datapoint *dp = malloc(sizeof(Mnist_Datapoint));
    dp->label = new_mat(1, 10); // one-hot
    dp->data = new_mat(1, 28 * 28);
    return dp;
}

int read_next_dp_from_file(FILE *file, Mnist_Datapoint *dp)
{
    char *line = read_next_line(file);
    if (line == NULL)
    {
        return 0;
    }
    parse_line_to_mat(line, dp->data, dp->label);
    return 1;
}
Mnist_Datapoint *get_next_datapoint(Mnist_Dataset *dataset)
{
    if (dataset->current_dp_index >= dataset->size_to_use)
    {
        dataset->current_dp_index = 0;
        return NULL;
    }
    Mnist_Datapoint *dp = dataset->datapoints[dataset->current_dp_index];
    dataset->current_dp_index++;
    return dp;
}

Mnist_Dataset *create_mnist_from_csv(char *path, int size_to_use)
{
    Mnist_Dataset *dataset = malloc(sizeof(Mnist_Dataset));
    dataset->size_to_use = size_to_use;
    dataset->current_dp_index = 0;
    dataset->path = path;
    dataset->datapoints = calloc(dataset->size_to_use, sizeof(Mnist_Datapoint *));

    FILE *file = fopen(path, "r");
    read_next_line(file); // skip first line

    for (int i = 0; i < dataset->size_to_use; ++i)
    {
        Mnist_Datapoint *dp = create_datapoint();
        dataset->datapoints[i] = dp;
        if (!read_next_dp_from_file(file, dp))
            break;
    }
    fclose(file);
    return dataset;
}

void free_datapoint(Mnist_Datapoint *dp)
{
    free_mat(dp->data);
    free_mat(dp->label);
    free(dp);
}

void free_dataset(Mnist_Dataset *dataset)
{
    for (int i = 0; i < dataset->size_to_use; i++)
    {
        free_datapoint(dataset->datapoints[i]);
    }
    free(dataset->datapoints);
    free(dataset);
}