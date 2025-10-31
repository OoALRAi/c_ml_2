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
        return NULL;
    }
    return line;
}

void one_hot_label(int label, Matrix *result)
{
    for (int i = 0; i < result->rows; i++)
    {
        set_element_at(result, 0, i, i == label);
    }
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
    float label_value = atof(current_char);
    one_hot_label(label_value, label);
    current_char = skip_value(current_char);
    for (int i = 0; i < 28 * 28; i++)
    {
        float value = atof(current_char);
        set_element_at(data, i, 0, value);
        current_char = skip_value(current_char);
        if (current_char == NULL)
            break;
    }
}

Mnist_Datapoint *create_datapoint()
{
    Mnist_Datapoint *dp = malloc(sizeof(Mnist_Datapoint));
    dp->label = new_mat(10, 1); // one-hot
    dp->data = new_mat(1, 28 * 28);
    return dp;
}

Mnist_Datapoint *mnist_next_datapoint(Mnist_Dataset *dataset)
{
    if (dataset == NULL)
    {
        fprintf(stderr, "dataset is null\n");
        exit(0);
    }
    if (dataset->file == NULL)
    {
        dataset->file = fopen(dataset->path, "r");
    }
    read_next_line(dataset->file); // skip legend
    char *line_data = read_next_line(dataset->file);
    if (line_data == NULL)
    {
        return NULL;
    }
    if (dataset->current_datapoint == NULL)
    {
        dataset->current_datapoint = create_datapoint();
    }
    parse_line_to_mat(line_data, dataset->current_datapoint->data, dataset->current_datapoint->label);

    return dataset->current_datapoint;
}

Mnist_Dataset *create_mnist_from_csv(char *path)
{
    Mnist_Dataset *dataset = malloc(sizeof(Mnist_Dataset));
    dataset->path = path;
    dataset->file = NULL;
    dataset->current_datapoint = NULL;
    return dataset;
}

void free_dataset(Mnist_Dataset *dataset)
{
    fclose(dataset->file);
    // TODO: free other resources
}

void free_dataopint(Mnist_Datapoint *dp)
{
    // TODO: free datapoint
}
