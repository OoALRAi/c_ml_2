#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

Matrix *new_mat(int rows, int cols)
{
    Matrix *m = malloc(sizeof(Matrix));
    m->rows = rows;
    m->cols = cols;
    m->data = calloc(rows * cols, sizeof(double));
    if (!m->data)
    {
        free(m->data);
        free(m);
        return NULL;
    }
    return m;
}

Matrix *new_mat_like(Matrix *m)
{
    if (!m)
    {
        fprintf(stderr, "[new_mat_like]: m is null\n");
        exit(0);
    }
    return new_mat(m->rows, m->cols);
}

Matrix *new_copy_of(Matrix *m)
{
    Matrix *cp_mat = new_mat_like(m);
    copy_mat(m, cp_mat);
    return cp_mat;
}

void free_mat(Matrix *m)
{
    if (m == NULL)
    {
        return;
    }
    if (m->data != NULL)
    {
        free(m->data);
    }
    free(m);
    m = NULL;
}

Matrix *zeros(int rows, int cols)
{
    Matrix *r = new_mat(rows, cols);
    const_fill_mat(0, r);
    return r;
}

Matrix *ones(int rows, int cols)
{
    Matrix *r = new_mat(rows, cols);
    const_fill_mat(1, r);
    return r;
}
int check_sizes(Matrix *m1, Matrix *m2)
{
    /*
    check if given matrices are of same sizes
    return: 1 if same sizes, 0 if different sizes
    */
    return m1 != NULL && m2 != NULL && m1->cols == m2->cols && m1->rows == m2->rows;
}
int check_sizes_for_dot(Matrix *m1, Matrix *m2)
{
    /*
    check if sizes of given matrices are appropriate for dot product;
    return: 1 if appropriate, else 0
    */
    return m1 != NULL && m2 != NULL && m1->cols == m2->rows;
}

void add_mat_to(Matrix *a, Matrix *b, Matrix *result)
{
    int equal_sizes = check_sizes(a, b);
    if (equal_sizes == 0)
    {
        fprintf(stderr, "[%s] matrices have different sizes or are null\n", __FUNCTION__);
        exit(-1);
    }

    for (int y = 0; y < a->rows; y++)
    {
        for (int x = 0; x < a->cols; x++)
        {
            SET_ELEMENT_AT(result, x, y, GET_ELEMENT_AT(a, x, y) + GET_ELEMENT_AT(b, x, y));
        }
    }
}

void sub_mat_to(Matrix *a, Matrix *b, Matrix *result)
{
    int equal_sizes = check_sizes(a, b);
    if (equal_sizes == 0)
    {
        fprintf(stderr, "[%s] matrices have different sizes or are null\n", __FUNCTION__);
        exit(-1);
    }

    for (int y = 0; y < a->rows; y++)
    {
        for (int x = 0; x < a->cols; x++)
        {
            SET_ELEMENT_AT(result, x, y, GET_ELEMENT_AT(a, x, y) - GET_ELEMENT_AT(b, x, y));
        }
    }
}

void dot_to(Matrix *a, Matrix *b, Matrix *result)

{
    int check_results = check_sizes_for_dot(a, b);
    if (check_results == 0)
    {
        fprintf(stderr, "[%s] dot product not possible\n", __FUNCTION__);
        exit(-1);
    }
    if (result == NULL)
    {
        fprintf(stderr, "[%s] matrix result is null\n", __FUNCTION__);
        exit(-1);
    }
    if (result->rows != a->rows || result->cols != b->cols)
    {

        fprintf(stderr, "[%s] dim of result matrix is invalid, expected dim result: (%d,%d), actual dim result: (%d,%d)\n",
                __FUNCTION__,
                a->rows, b->cols, result->rows, result->cols);
        exit(-1);
    }
    for (int y = 0; y < a->rows; y++)
    {
        for (int x = 0; x < b->cols; x++)
        {
            double acc = 0;
            for (int i = 0; i < a->cols; i++)
            {
                acc += GET_ELEMENT_AT(a, i, y) * GET_ELEMENT_AT(b, x, i);
            }
            SET_ELEMENT_AT(result, x, y, acc);
        }
    }
}

void e_div_mat_to(Matrix *a, Matrix *b, Matrix *result)
{
    if (a == NULL)
    {
        fprintf(stderr, "matrix a is null\n");
        exit(-1);
    }
    if (b == NULL)
    {
        fprintf(stderr, "matrix b is null\n");
        exit(-1);
    }
    if (result == NULL)
    {
        fprintf(stderr, "result matrix is null\n");
        exit(-1);
    }
    if (a->cols != b->cols || a->rows != b->rows)
    {
        fprintf(stderr, "input matrices a and b have different dimensions\n");
        exit(-1);
    }
    if (a->cols != result->cols || a->rows != result->rows)
    {
        fprintf(stderr, "incorrect output dimensions\n");
        exit(-1);
    }
    for (int y = 0; y < a->rows; y++)
    {
        for (int x = 0; x < a->cols; x++)
        {
            SET_ELEMENT_AT(result, x, y, GET_ELEMENT_AT(a, x, y) / GET_ELEMENT_AT(b, x, y));
        }
    }
}

void e_mul_mat_to(Matrix *a, Matrix *b, Matrix *result)
{
    {
        if (a == NULL)
        {
            fprintf(stderr, "matrix a is null\n");
            exit(-1);
        }
        if (b == NULL)
        {
            fprintf(stderr, "matrix b is null\n");
            exit(-1);
        }
        if (result == NULL)
        {
            fprintf(stderr, "result matrix is null\n");
            exit(-1);
        }
        if (a->cols != b->cols || a->rows != b->rows)
        {
            fprintf(stderr, "input matrices a and b have different dimensions\n");
            exit(-1);
        }
        if (a->cols != result->cols || a->rows != result->rows)
        {
            fprintf(stderr, "incorrect output dimensions\n");
            exit(-1);
        }
        for (int y = 0; y < a->rows; y++)
        {
            for (int x = 0; x < a->cols; x++)
            {
                SET_ELEMENT_AT(result, x, y, GET_ELEMENT_AT(a, x, y) * GET_ELEMENT_AT(b, x, y));
            }
        }
    }
}

void transpose_mat_to(Matrix *m, Matrix *result)
{
    if (m == NULL)
    {
        fprintf(stderr, "input matrix is null\n");
        exit(-1);
    }
    if (result == NULL)
    {
        fprintf(stderr, "output matrix is null\n");
        exit(-1);
    }
    if (m->cols != result->rows || m->rows != result->cols)
    {
        fprintf(stderr, "incorrect output dimenstions, expexted dim: (%dx%d), actual dim: (%dx%d)\n",
                m->cols, m->rows, result->rows, result->cols);
        exit(-1);
    }
    for (size_t y = 0; y < m->rows; y++)
    {
        for (size_t x = 0; x < m->cols; x++)
        {
            SET_ELEMENT_AT(result, y, x, GET_ELEMENT_AT(m, x, y));
        }
    }
}

void e_pow_mat_to(Matrix *m, Matrix *result, double pow_value)
{
    if (m == NULL)
    {
        fprintf(stderr, "m is null\n");
        exit(-1);
    }
    if (result == NULL)
    {
        fprintf(stderr, "result matrix is null\n");
        exit(-1);
    }
    if (m->rows != result->rows || m->cols != result->cols)
    {
        fprintf(stderr, "dimensions of input and result matrices do\'t match\n");
        exit(-1);
    }

    for (size_t y = 0; y < m->rows; y++)
    {
        for (size_t x = 0; x < m->cols; x++)
        {
            double value = GET_ELEMENT_AT(m, x, y);
            SET_ELEMENT_AT(
                result, x, y,
                pow(value, pow_value));
        }
    }
}

void transpose_mat_inplace(Matrix *m)
{
    if (m == NULL)
    {
        fprintf(stderr, "matrix is null\n");
        exit(-1);
    }
    if (m->cols != m->rows)
    {
        fprintf(stderr, "m is not quadratic, inplace transpose is only for quadratic matrix possible\n");
        exit(-1);
    }
    int x_start = 0;
    for (int y = 0; y < m->rows; y++)
    {
        for (int x = x_start; x < m->cols; x++)
        {
            double temp = m->data[y * m->cols + x];
            m->data[y * m->cols + x] = m->data[x * m->cols + y];
            m->data[x * m->cols + y] = temp;
        }
        x_start++;
    }
    int temp_cols = m->cols;
    m->cols = m->rows;
    m->rows = temp_cols;
}

void scale_mat_inplace(Matrix *m, double scaler)
{
    if (m == NULL)
    {
        fprintf(stderr, "input matrix is null\n");
        exit(-1);
    }
    for (size_t y = 0; y < m->rows; y++)
    {

        for (size_t x = 0; x < m->cols; x++)
        {
            SET_ELEMENT_AT(m, x, y, GET_ELEMENT_AT(m, x, y) * scaler);
        }
    }
}

Matrix *add_mat(Matrix *a, Matrix *b)
{
    Matrix *r = new_mat(a->rows, a->cols);
    add_mat_to(a, b, r);
    return r;
}

Matrix *sub_mat(Matrix *a, Matrix *b)
{
    Matrix *r = new_mat(a->rows, a->cols);
    sub_mat_to(a, b, r);
    return r;
}

Matrix *mul_mat(Matrix *a, Matrix *b)
{
    Matrix *r = new_mat(a->rows, b->cols);
    dot_to(a, b, r);
    return r;
}

Matrix *elementwise_div_mat(Matrix *a, Matrix *b)
{
    Matrix *r = new_mat(a->rows, a->cols);
    e_div_mat_to(a, b, r);
    return r;
}

void div_mat_by_value_to(Matrix *m, double value, Matrix *result)
{
    if (m == NULL)
    {
        fprintf(stderr, "m is null\n");
        exit(-1);
    }
    if (result == NULL)
    {
        fprintf(stderr, "result matrix is null\n");
        exit(-1);
    }
    if (m->rows != result->rows || m->cols != result->cols)
    {
        fprintf(stderr, "dimensions of input and result matrices do\'t match\n");
        exit(-1);
    }

    for (size_t y = 0; y < m->rows; y++)
    {
        for (size_t x = 0; x < m->cols; x++)
        {
            SET_ELEMENT_AT(
                result, x, y,
                GET_ELEMENT_AT(m, x, y) / value);
        }
    }
}

Matrix *div_mat_by_value(Matrix *m, double value)
{
    if (m == NULL)
    {
        fprintf(stderr, "m is null\n");
        exit(-1);
    }
    Matrix *result = new_mat(m->rows, m->cols);
    div_mat_by_value_to(m, value, result);
    return result;
}
void scale_mat_to(Matrix *m, double scaler, Matrix *result)
{
    if (m == NULL)
    {
        fprintf(stderr, "m is null\n");
        exit(-1);
    }
    if (result == NULL)
    {
        fprintf(stderr, "result matrix is null\n");
        exit(-1);
    }
    if (m->rows != result->rows || m->cols != result->cols)
    {
        fprintf(stderr, "dimensions of input and result matrices do\'t match\n");
        exit(-1);
    }

    for (size_t y = 0; y < m->rows; y++)
    {
        for (size_t x = 0; x < m->cols; x++)
        {
            SET_ELEMENT_AT(
                result, x, y,
                GET_ELEMENT_AT(m, x, y) * scaler);
        }
    }
}

Matrix *elementwise_mul_mat(Matrix *a, Matrix *b)
{

    Matrix *r = new_mat(a->rows, a->cols);
    e_mul_mat_to(a, b, r);
    return r;
}

Matrix *elementwise_pow_mat(Matrix *m, double pow_value)
{
    if (m == NULL)
    {
        fprintf(stderr, "m is null\n");
        exit(-1);
    }
    Matrix *result = new_mat(m->rows, m->cols);
    e_pow_mat_to(m, result, pow_value);
    return result;
}

Matrix *transpose_mat(Matrix *m)
{
    Matrix *r = new_mat(m->cols, m->rows);
    transpose_mat_to(m, r);
    return r;
}

double max(Matrix *m)
{
    if (m == NULL)
    {
        fprintf(stderr, "matrix is null\n");
        exit(0);
    }
    double max = -INFINITY;
    for (int y = 0; y < m->rows; y++)
    {
        for (int x = 0; x < m->cols; x++)
        {
            double v = GET_ELEMENT_AT(m, x, y);
            max = v > max ? v : max;
        }
    }
    return max;
}
int argmax(Matrix *m)
{
    if (m == NULL)
    {
        fprintf(stderr, "matrix is null\n");
        exit(0);
    }
    double max = -INFINITY;
    int arg = -1;
    for (int x = 0; x < m->cols; x++)
    {
        double v = GET_ELEMENT_AT(m, x, 0);
        if (v > max)
        {
            max = v;
            arg = x;
        }
    }
    return arg;
}

void print_mat(Matrix *m)
{
    if (m == NULL)
    {
        fprintf(stderr, "matrix is null\n");
        exit(-1);
    }
    if (m->data == NULL)
    {
        fprintf(stderr, "data of the matrix is null\n");
        exit(-1);
    }

    printf("(%d x %d)\n", m->rows, m->cols);
    printf("[\n");
    for (int y = 0; y < m->rows; y++)
    {
        for (int x = 0; x < m->cols; x++)
        {
            printf("\t%.2f ", m->data[y * m->cols + x]);
        }
        printf("\n");
    }
    printf("]\n");
}

void copy_mat(Matrix *source, Matrix *target)
{
    if (source == NULL)
    {
        fprintf(stderr, "source matrix is null\n");
        exit(-1);
    }
    if (target == NULL)
    {
        fprintf(stderr, "target matrix is null\n");
        exit(-1);
    }
    if (source->rows != target->rows || source->cols != target->cols)
    {
        fprintf(stderr, "dimensions of source and target matrices do not match\n");
        exit(-1);
    }
    for (size_t y = 0; y < source->rows; y++)
    {

        for (size_t x = 0; x < source->cols; x++)
        {
            SET_ELEMENT_AT(target, x, y, GET_ELEMENT_AT(source, x, y));
        }
    }
}

void const_fill_mat(double value, Matrix *m)
{
    if (m == NULL)
    {
        printf("cannot fill null matrix\n");
        return;
    }
    else if (m->data == NULL)
    {
        printf("cannot fill a matrix with null data\n");
        return;
    }

    for (int y = 0; y < m->rows; y++)
    {
        for (int x = 0; x < m->cols; x++)
        {
            m->data[y * m->cols + x] = value;
        }
    }
}

void stepwise_fill_mat(double start, double step, Matrix *m)
{
    if (m == NULL)
    {
        printf("cannot fill null matrix\n");
        return;
    }
    else if (m->data == NULL)
    {
        printf("cannot fill a matrix with null data\n");
        return;
    }

    for (int y = 0; y < m->rows; y++)
    {
        for (int x = 0; x < m->cols; x++)
        {
            m->data[y * m->cols + x] = start;
            start += step;
        }
    }
}

double get_random_number()
{
    static int SEED_INITIALIZED = 0;
    if (SEED_INITIALIZED == 0)
    {
        srand(time(NULL));
        SEED_INITIALIZED = 1;
    }
    double value = (((double)rand()) / RAND_MAX) * 2 - 1;
    return value;
}
void random_fill_mat(Matrix *m)
{
    if (m == NULL)
    {
        printf("input matrix is null\n");
        exit(-1);
    }
    for (size_t y = 0; y < m->rows; y++)
    {

        for (size_t x = 0; x < m->cols; x++)
        {
            SET_ELEMENT_AT(m, x, y, get_random_number());
        }
    }
}

Matrix *random_mat(int rows, int cols)
{
    Matrix *m = new_mat(rows, cols);
    random_fill_mat(m);
    return m;
}