#include "statistic_utils.h"
#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>

Confusion_Matrix *create_confision_matrix(int num_classes)
{
    Confusion_Matrix *cm = malloc(sizeof(Confusion_Matrix));

    int shape[] = {num_classes, num_classes};
    cm->current_cm = new_mat(shape, 2);
    cm->previous_cm = new_mat(shape, 2);
    return cm;
}

void add_prediction(Confusion_Matrix *cm, int gt, int pred)
{
    // Matrix inside Confusion_Matrix is assumed to be 2D
    int stride = cm->current_cm->stride[0];
    double old_value = cm->current_cm->data[gt * stride + pred];
    cm->current_cm->data[gt * stride + pred] = old_value + 1;
}

void end_epoch(Confusion_Matrix *cm)
{
    copy_mat(cm->current_cm, cm->previous_cm);
    fill_mat_with(0, cm->current_cm);
}

int get_TP_of(Matrix *cm, int cls)
{
    return cm->data[cls * cm->stride[0] + cls];
}

int get_FP_of(Matrix *cm, int cls)
{
    int fp = 0;
    for (int y = 0; y < cm->shape[0]; y++)
    {
        if (y == cls)
            continue;
        int value = (int)cm->data[y * cm->stride[0] + cls];
        fp += value;
    }
    return fp;
}
int get_FN_of(Matrix *cm, int cls)
{
    int fn = 0;
    for (int x = 0; x < cm->shape[1]; x++)
    {
        if (x == cls)
            continue;
        // int value = (int)GET_ELEMENT_AT(cm, x, cls);
        int value = (int)cm->data[cls * cm->stride[0] + x];
        fn += value;
    }
    return fn;
}

Matrix *compute_precision(Matrix *cm)
{
    int shape[] = {1, cm->shape[1]};
    Matrix *precisions = new_mat(shape, 2);
    for (int cls = 0; cls < cm->shape[1]; cls++)
    {
        int tp = get_TP_of(cm, cls);
        int fp = get_FP_of(cm, cls);
        double denominator = tp + fp;
        if (denominator == 0)
        {
            // avoid divide by 0
            precisions->data[0] = cls;
        }
        else
        {
            double precision_cls = (double)tp / denominator;
            // SET_ELEMENT_AT(precisions, cls, 0, precision_cls);
            precisions->data[cls] = precision_cls;
        }
    }
    return precisions;
}

Matrix *compute_recall(Matrix *cm)
{
    int shape[] = {1, cm->shape[1]};
    Matrix *recalls = new_mat(shape, 2);
    for (int cls = 0; cls < cm->shape[1]; cls++)
    {
        int tp = get_TP_of(cm, cls);
        int fn = get_FN_of(cm, cls);
        double denominator = tp + fn;
        if (denominator == 0)
        {
            // avoid divide by 0
            recalls->data[0] = cls;
        }
        else
        {
            double recall_cls = (double)tp / denominator;
            recalls->data[cls] = recall_cls;
        }
    }
    return recalls;
}

double compute_f1(double precision, double recall)
{
    if ((precision + recall) == 0)
    {
        return 0;
    }
    return 2 * precision * recall / (precision + recall);
}

char get_stat_symbol(double current, double prev)
{
    if (current > prev)
        return '+';
    else if (current < prev)
        return '-';
    else
        return '=';
}

void print_stats(Confusion_Matrix *cm)
{
    printf("class\tP\tR\tf1\n");
    Matrix *p = compute_precision(cm->current_cm);
    Matrix *r = compute_recall(cm->current_cm);

    Matrix *prev_p = compute_precision(cm->previous_cm);
    Matrix *prev_r = compute_recall(cm->previous_cm);

    for (int cls = 0; cls < cm->current_cm->shape[1]; cls++)
    {
        // current stats
        double p_cls = p->data[cls];              // precision of cls
        double r_cls = r->data[cls];              // recall of cls
        double f1_cls = compute_f1(p_cls, r_cls); // f1 of cls

        // previous stats
        double prev_p_cls = prev_p->data[cls];
        double prev_r_cls = prev_r->data[cls];
        double prev_f1_cls = compute_f1(prev_p_cls, prev_r_cls);

        char p_symbol, r_symbol, f1_symbol;
        p_symbol = get_stat_symbol(p_cls, prev_p_cls);
        r_symbol = get_stat_symbol(r_cls, prev_r_cls);
        f1_symbol = get_stat_symbol(f1_cls, prev_f1_cls);

        printf("%d\t%.2f(%c)\t%.2f(%c)\t%.2f(%c)\n",
               cls,
               p_cls, // precision of cls
               p_symbol,
               r_cls, // recall of cls
               r_symbol,
               f1_cls,
               f1_symbol);
    }
    free_mat(p);
    free_mat(r);
}
void print_confusion_mat(Confusion_Matrix *cm)
{
    printf("\t\tprediction\n");
    printf("\t\t");
    for (int cls = 0; cls < cm->current_cm->shape[1]; cls++)
    {
        printf("%d\t", cls);
    }
    printf("\n");
    printf("ground truth\n");

    for (int cls = 0; cls < cm->current_cm->shape[0]; cls++)
    {
        printf("class %d\t", cls);
        for (int i = 0; i < cm->current_cm->shape[1]; i++)
        {
            double value = cm->current_cm->data[cls * cm->current_cm->stride[0] + i];
            printf("\t%0.f", value);
        }
        printf("\n");
    }
}
