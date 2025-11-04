#ifndef STATS_H
#define STATS_H
#include "matrix.h"

typedef struct
{
    Matrix *current_cm;
    Matrix *previous_cm;
} Confusion_Matrix;

Confusion_Matrix *create_confision_matrix(int num_classes);
void add_prediction(Confusion_Matrix *cm, int gt, int pred);
void end_epoch(Confusion_Matrix *cm);

int get_TP_of(Matrix *cm, int cls);
int get_FP_of(Matrix *cm, int cls);
int get_FN_of(Matrix *cm, int cls);

Matrix *compute_precision(Matrix *cm);
Matrix *compute_recall(Matrix *cm);
double compute_f1(double precision, double recall);
char get_stat_symbol(double current, double prev);
void print_stats(Confusion_Matrix *cm);
void print_confusion_mat(Confusion_Matrix *cm);

#endif