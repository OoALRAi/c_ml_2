#include "statistic_utils.h"
#include "matrix.h"
#include <stdio.h>

Matrix *create_confision_matrix(int num_classes)
{
    return new_mat(num_classes, num_classes);
}

void add_prediction(Matrix *cm, int gt, int pred)
{
    double old_value = get_element_at(cm, pred, gt);
    set_element_at(cm, pred, gt, old_value + 1);
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
        double precision_cls = (double)tp / (double)(tp + fp);
        set_element_at(precisions, cls, 0, precision_cls);
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
        double recall_cls = (double)tp / (double)(tp + fn);
        set_element_at(recalls, cls, 0, recall_cls);
    }
    return recalls;
}
void print_stats(Matrix *cm)
{
    printf("class\tP\tR\n");
    Matrix *p = compute_precision(cm);
    Matrix *r = compute_recall(cm);
    for (int cls = 0; cls < cm->cols; cls++)
    {
        printf("%d\t%.2f\t%.2f\n", cls, get_element_at(p, cls, 0), get_element_at(r, cls, 0));
    }
}
void print_confusion_mat(Matrix *cm)
{
    printf("\t\tprediction\n");
    printf("\t\t");
    for (int cls = 0; cls < cm->cols; cls++)
    {
        printf("%d\t", cls);
    }
    printf("\n");
    printf("ground truth\n");

    for (int cls = 0; cls < cm->rows; cls++)
    {
        printf("class %d\t", cls);
        for (int i = 0; i < cm->cols; i++)
        {
            printf("\t%0.f", get_element_at(cm, i, cls));
        }
        printf("\n");
    }
}
