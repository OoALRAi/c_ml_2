#ifndef MATRIX_H
#define MATRIX_H

#define GET_ELEMENT_AT(m, x, y) (m->data[y * (m->stride) + x])
#define SET_ELEMENT_AT(m, x, y, v) (m->data[y * (m->stride) + x] = v)

typedef struct Matrix
{
    int owner;
    int ndims;  // size of shape list
    int *shape; // list of dimensions sizes
    // for e.g. 2x3 matrix has the shape [2, 3]
    // an 28x28 rgb image with has 3 channels,
    // thus the shape is [3, 28, 28]
    // i.e. 3 channels and each of size 28 x 28

    int *stride; // size of strides = ndims - 1

    double *data;
    int size; // size of data array
} Matrix;

Matrix *new_mat(int *shape, int ndims);
Matrix *new_view(int *shape, int ndims, int *stride);
Matrix *new_mat_like(Matrix *m);
Matrix *new_copy_of(Matrix *m);
void free_mat(Matrix *m);
void print_mat(Matrix *m);
int equals(Matrix *m1, Matrix *m2);

Matrix *zeros(int *shape, int ndims);
Matrix *ones(int *shape, int ndims);

int check_shapes_elementweise_op(Matrix *m1, Matrix *m2);
int check_shapes_for_mul_mat(Matrix *m1, Matrix *m2, Matrix *result);

// ==== no memory allocation in these operations ====
void add_mat_to(Matrix *a, Matrix *b, Matrix *result);
void sub_mat_to(Matrix *a, Matrix *b, Matrix *result);

void e_div_mat_to(Matrix *a, Matrix *b, Matrix *result);
void div_mat_by_value_to(Matrix *m, double value, Matrix *result);
void e_mul_mat_to(Matrix *a, Matrix *b, Matrix *result);
void e_pow_mat_to(Matrix *m, Matrix *result, double pow_value);
void scale_mat_to(Matrix *m, double scaler, Matrix *result);
void scale_mat_inplace(Matrix *m, double scaler);

double max(Matrix *m);
int argmax(Matrix *m);
void squeeze_first_dim(Matrix *m);

/**
 * Compute dot product of input matrices.
 * For 1D input matrices compute dot product directly.
 * For 2D input matrices compute the inner product of them
 * which is like sum of element-wise product of values of
 * the two input matrices.
 * @param a: Matrix pointer
 * @param b: Matrix pointer
 * @return double
 */
double dot_mat(Matrix *a, Matrix *b);
Matrix *slice_mat(Matrix *m, int *slice_range, int slice_range_size);
// ==== no memory allocation in these operations ====

// ==== memory allocation ====
Matrix *add_mat(Matrix *a, Matrix *b);
Matrix *sub_mat(Matrix *a, Matrix *b);
Matrix *div_mat_by_value(Matrix *m, double value);
void transpose_mat(Matrix *m, int dim1, int dim2);
// ==== memory allocation ====

// help functions to fill matrix
void copy_mat(Matrix *source, Matrix *target);
void fill_mat_with(double value, Matrix *m);
void stepwise_fill_mat(double start, double step, Matrix *m);
Matrix *random_mat(int *shape, int ndims);
void random_fill_mat(Matrix *m);
double get_random_number();
#endif