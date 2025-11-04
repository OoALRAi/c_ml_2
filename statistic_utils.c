#include "statistic_utils.h"
#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>

Confusion_Matrix *create_confision_matrix(int num_classes)
{
    Confusion_Matrix *cm = malloc(sizeof(Confusion_Matrix));
    cm->current_cm = new_mat(num_classes, num_classes);
    cm->previous_cm = new_mat(num_classes, num_classes);
    return cm;
}

void add_prediction(Confusion_Matrix *cm, int gt, int pred)
{
    double old_value = get_element_at(cm->current_cm, pred, gt);
    set_element_at(cm->current_cm, pred, gt, old_value + 1);
}

void end_epoch(Confusion_Matrix *cm)
{
    copy_mat(cm->current_cm, cm->previous_cm);
    const_fill_mat(0, cm->current_cm);
}

int get_TP_of(Matrix *cm, int cls)
{
    return get_element_at(cm, cls, cls);
}

int get_FP_of(Matrix *cm, int cls)
{
    int fp = 0;
    for (int y = 0; y < cm->rows; y++)
    {
        if (y == cls)
            continue;
        int value = (int)get_element_at(cm, cls, y);
        fp += value;
    }
    return fp;
}
int get_FN_of(Matrix *cm, int cls)
{
    int fn = 0;
    for (int x = 0; x < cm->cols; x++)
    {
        if (x == cls)
            continue;
        int value = (int)get_element_at(cm, x, cls);
        fn += value;
    }
    return fn;
}

Matrix *compute_precision(Matrix *cm)
{
    Matrix *precisions = new_mat(1, cm->cols);
    for (int cls = 0; cls < cm->cols; cls++)
    {
        int tp = get_TP_of(cm, cls);
        int fp = get_FP_of(cm, cls);
        double denominator = tp + fp;
        if (denominator == 0)
        {
            // avoid divide by 0
            set_element_at(precisions, cls, 0, 0);
        }
        else
        {
            double precision_cls = (double)tp / denominator;
            set_element_at(precisions, cls, 0, precision_cls);
        }
    }
    return precisions;
}

Matrix *compute_recall(Matrix *cm)
{
    Matrix *recalls = new_mat(1, cm->cols);
    for (int cls = 0; cls < cm->cols; cls++)
    {
        int tp = get_TP_of(cm, cls);
        int fn = get_FN_of(cm, cls);
        double denominator = tp + fn;
        if (denominator == 0)
        {
            // avoid divide by 0
            set_element_at(recalls, cls, 0, 0);
        }
        else
        {
            double recall_cls = (double)tp / denominator;
            set_element_at(recalls, cls, 0, recall_cls);
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

    for (int cls = 0; cls < cm->current_cm->cols; cls++)
    {
        // current stats
        double p_cls = get_element_at(p, cls, 0); // precision of cls
        double r_cls = get_element_at(r, cls, 0); // recall of cls
        double f1_cls = compute_f1(p_cls, r_cls); // f1 of cls

        // previous stats
        double prev_p_cls = get_element_at(prev_p, cls, 0);
        double prev_r_cls = get_element_at(prev_r, cls, 0);
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
    for (int cls = 0; cls < cm->current_cm->cols; cls++)
    {
        printf("%d\t", cls);
    }
    printf("\n");
    printf("ground truth\n");

    for (int cls = 0; cls < cm->current_cm->rows; cls++)
    {
        printf("class %d\t", cls);
        for (int i = 0; i < cm->current_cm->cols; i++)
        {
            printf("\t%0.f", get_element_at(cm->current_cm, i, cls));
        }
        printf("\n");
    }
}
