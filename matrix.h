#ifndef MATRIX_H
#define MATRIX_H

typedef struct Matrix
{
    int rows;
    int cols;
    double *data;
} Matrix;

Matrix *new_mat(int rows, int cols);
void free_mat(Matrix *m);
void print_mat(Matrix *m);

Matrix *zeros(int rows, int cols);
Matrix *ones(int rows, int cols);

double get_element_at(Matrix *m, int x, int y);
void set_element_at(Matrix *m, int x, int y, double value);

// these operation create no new result matrix but stroe the result to
// already allocated matrix
void add_mat_to(Matrix *a, Matrix *b, Matrix *result);
void sub_mat_to(Matrix *a, Matrix *b, Matrix *result);
void mul_mat_to(Matrix *a, Matrix *b, Matrix *result);

void elementwise_div_mat_to(Matrix *a, Matrix *b, Matrix *result);
void div_mat_by_value_to(Matrix *m, Matrix *result, double value);
void elementwise_mul_mat_to(Matrix *a, Matrix *b, Matrix *result);
void transpose_mat_to(Matrix *m, Matrix *result);
void elementwise_pow_mat_to(Matrix *m, Matrix *result, double pow_value);
void multiply_mat_with_value_to(Matrix *m, Matrix *result, double value);

void transpose_mat_inplace(Matrix *m);

void scale_mat_inplace(Matrix *m, double scaler);

// these operation allocate new result matrix and return it.
Matrix *add_mat(Matrix *a, Matrix *b);
Matrix *sub_mat(Matrix *a, Matrix *b);
Matrix *mul_mat(Matrix *a, Matrix *b);
Matrix *elementwise_div_mat(Matrix *a, Matrix *b);
Matrix *div_mat_by_value(Matrix *m, double value);
Matrix *elementwise_mul_mat(Matrix *a, Matrix *b);
Matrix *elementwise_pow_mat(Matrix *m, double pow_value);
Matrix *transpose_mat(Matrix *m);
double max(Matrix *m);
int max_arg(Matrix *m);

// help functions to fill matrix
void copy_mat(Matrix *source, Matrix *target);
void const_fill_mat(double value, Matrix *m);
void stepwise_fill_mat(double start, double step, Matrix *m);
Matrix *random_mat(int rows, int cols);
void random_fill_mat(Matrix *m);
double get_random_number();
#endif