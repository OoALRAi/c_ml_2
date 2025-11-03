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
        set_element_at(result, i, 0, i == label);
    }
}
int label_from_one_hot(Matrix *one_hot_label)
{
    int label = -1;
    for (int i = 0; i < one_hot_label->cols; i++)
    {
        if (get_element_at(one_hot_label, i, 0) == 1)
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

        set_element_at(data, i, 0, value);
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
void reset_dataset(Mnist_Dataset *ds)
{
    ds->used_datapoints = 0;
    fclose(ds->file);
    ds->file = NULL;
}

/**
 * returns next datapoint in mnist dataset
 * if the dataset has no more datapoints it resets and returns NULL
 * the next time calling this function will return the first datapoint again.
 */
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
        char *next_line = read_next_line(dataset->file); // skip legend
        free(next_line);
    }
    char *line_data = read_next_line(dataset->file);
    if (line_data == NULL || dataset->used_datapoints >= dataset->size_to_use)
    {
        reset_dataset(dataset);
        return NULL;
    }
    if (dataset->current_datapoint == NULL)
    {
        dataset->current_datapoint = create_datapoint();
    }
    parse_line_to_mat(line_data, dataset->current_datapoint->data, dataset->current_datapoint->label);
    free(line_data);
    dataset->used_datapoints++;

    return dataset->current_datapoint;
}

Mnist_Dataset *create_mnist_from_csv(char *path, int size_to_use)
{
    Mnist_Dataset *dataset = malloc(sizeof(Mnist_Dataset));
    dataset->size_to_use = size_to_use;
    dataset->used_datapoints = 0;
    dataset->path = path;
    dataset->file = NULL;
    dataset->current_datapoint = NULL;
    return dataset;
}

void free_dataset(Mnist_Dataset *dataset)
{
    fclose(dataset->file);
    if (dataset->current_datapoint)
    {
        free_dataopint(dataset->current_datapoint);
    }
    free(dataset);
    // TODO: free other resources
}

void free_dataopint(Mnist_Datapoint *dp)
{

    free_mat(dp->data);
    free_mat(dp->label);
    free(dp);
    // TODO: free datapoint
}
