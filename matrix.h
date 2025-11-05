#ifndef MATRIX_H
#define MATRIX_H

#define GET_ELEMENT_AT(m, x, y) (m->data[y * (m->cols) + x])
#define SET_ELEMENT_AT(m, x, y, v) (m->data[y * (m->cols) + x] = v)

typedef struct Matrix
{
    int rows;
    int cols;
    double *data;
} Matrix;

Matrix *new_mat(int rows, int cols);
Matrix *new_mat_like(Matrix *m);
Matrix *new_copy_of(Matrix *m);
void free_mat(Matrix *m);
void print_mat(Matrix *m);

Matrix *zeros(int rows, int cols);
Matrix *ones(int rows, int cols);

int check_sizes(Matrix *m1, Matrix *m2);
int check_sizes_for_dot(Matrix *m1, Matrix *m2);

// ==== no memory allocation in these operations ====
void add_mat_to(Matrix *a, Matrix *b, Matrix *result);
void sub_mat_to(Matrix *a, Matrix *b, Matrix *result);
void dot_to(Matrix *a, Matrix *b, Matrix *result);

void e_div_mat_to(Matrix *a, Matrix *b, Matrix *result);
void div_mat_by_value_to(Matrix *m, double value, Matrix *result);
void e_mul_mat_to(Matrix *a, Matrix *b, Matrix *result);
void transpose_mat_to(Matrix *m, Matrix *result);
void e_pow_mat_to(Matrix *m, Matrix *result, double pow_value);
void scale_mat_to(Matrix *m, double scaler, Matrix *result);
void scale_mat_inplace(Matrix *m, double scaler);

void transpose_mat_inplace(Matrix *m);
double max(Matrix *m);
int argmax(Matrix *m);

double scalar_product(Matrix *a, Matrix *b);
Matrix *slice_mat(Matrix *m, int w_start, int w_end, int h_start, int h_end);
// ==== no memory allocation in these operations ====

// ==== memory allocation ====
Matrix *add_mat(Matrix *a, Matrix *b);
Matrix *sub_mat(Matrix *a, Matrix *b);
Matrix *mul_mat(Matrix *a, Matrix *b);
Matrix *e_div_mat(Matrix *a, Matrix *b);
Matrix *div_mat_by_value(Matrix *m, double value);
Matrix *e_mul_mat(Matrix *a, Matrix *b);
Matrix *e_pow_mat(Matrix *m, double pow_value);
Matrix *transpose_mat(Matrix *m);
// ==== memory allocation ====

// help functions to fill matrix
void copy_mat(Matrix *source, Matrix *target);
void fill_mat_with(double value, Matrix *m);
void stepwise_fill_mat(double start, double step, Matrix *m);
Matrix *random_mat(int rows, int cols);
void random_fill_mat(Matrix *m);
double get_random_number();
#endif