#ifndef STATS_H
#define STATS_H
#include "matrix.h"

Matrix *create_confision_matrix(int num_classes);
void add_prediction(Matrix *cm, int gt, int pred);

int get_TP_of(Matrix *cm, int cls);
int get_FP_of(Matrix *cm, int cls);
int get_FN_of(Matrix *cm, int cls);

Matrix *compute_precision(Matrix *cm);
Matrix *compute_recall(Matrix *cm);
void print_stats(Matrix *cm);
void print_confusion_mat(Matrix *cm);

#endif