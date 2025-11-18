#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <assert.h>

Matrix *new_mat(int *shape, int ndims)
{
    Matrix *m = malloc(sizeof(Matrix));

    m->owner = 1;
    m->ndims = ndims;
    m->shape = shape;

    int size = 1;
    for (int i = 0; i < ndims; i++)
    {
        size *= shape[i];
    }
    m->size = size;
    m->data = calloc(m->size, sizeof(double));
    if (!m->data)
    {
        free(m->data);
        free(m);
        return NULL;
    }

    // === compute stride array ===
    int *stride = malloc(ndims * sizeof(int));

    // stride of last dim is always 1
    // considering data array contains only double values
    stride[ndims - 1] = 1;
    for (int i = ndims - 1; i > 0; i--)
    {
        stride[i - 1] = stride[i] * shape[i];
    }
    m->stride = stride;

    // === compute stride array ===
    return m;
}

Matrix *new_view(int *shape, int ndims, int *stride)
{
    Matrix *m = malloc(sizeof(Matrix));
    m->ndims = ndims;
    m->shape = shape;
    m->stride = stride;
    m->owner = 0;
    m->data = NULL;
    return m;
}

Matrix *new_mat_like(Matrix *m)
{
    assert(m != NULL);
    assert(m->data != NULL);

    int ndims = m->ndims;
    int *new_mat_shape = malloc(m->ndims * sizeof(int));
    for (int i = 0; i < ndims; i++)
    {
        new_mat_shape[i] = m->shape[i];
    }
    Matrix *new_m = new_mat(new_mat_shape, ndims);
    return new_m;
}

Matrix *new_copy_of(Matrix *m)
{
    assert(m != NULL);
    assert(m->data != NULL);
    Matrix *cp_mat = new_mat_like(m);
    copy_mat(m, cp_mat);
    return cp_mat;
}

Matrix *zeros(int *shape, int ndims)
{
    Matrix *m = new_mat(shape, ndims);
    fill_mat_with(0, m);
    return m;
}

Matrix *ones(int *shape, int ndims)
{
    Matrix *m = new_mat(shape, ndims);
    fill_mat_with(1, m);
    return m;
}

void free_mat(Matrix *m)
{
    if (m == NULL)
    {
        return;
    }
    if (m->owner == 1)
    {
        free(m->data);
    }
    free(m);
    m = NULL;
}

int equals(Matrix *m1, Matrix *m2)
{
    // === ERROR CHECK ===
    assert(m1 != NULL);
    assert(m1->data != NULL);
    assert(m2 != NULL);
    assert(m2->data != NULL);
    // === ERROR CHECK ===

    if (m1->ndims != m2->ndims)
    {
        return 0;
    }
    for (int dim = 0; dim < m1->ndims; ++dim)
    {
        if (m1->shape[dim] != m2->shape[dim])
        {
            return 0;
        }
    }

    int ndims = m1->ndims;
    int *idx = calloc(ndims, sizeof(int)); // current indices

    for (int count = 0; count < m1->size; count++)
    {
        // Compute current offsets using stride and index counter
        int offset_m1 = 0, offset_m2 = 0;
        for (int d = 0; d < ndims; d++)
        {
            offset_m1 += idx[d] * m1->stride[d];
            offset_m2 += idx[d] * m2->stride[d];
        }

        double v1 = m1->data[offset_m1];
        double v2 = m2->data[offset_m2];
        if (v1 != v2)
            return 0;

        // compute next index
        for (int d = ndims - 1; d >= 0; d--)
        {
            idx[d]++;
            if (idx[d] < m1->shape[d])
                break;
            idx[d] = 0;
        }
    }
    free(idx);
    return 1;
}

int check_shapes_elementweise_op(Matrix *a, Matrix *b)
{
    if (a == NULL)
    {
        fprintf(stderr, "[%s] matrix a is null\n", __FUNCTION__);
        return 0;
    }
    if (b == NULL)
    {
        fprintf(stderr, "[%s] matrix a is null\n", __FUNCTION__);
        return 0;
    }
    int equal_sizes = 1;
    if (a->ndims != b->ndims)
    {
        equal_sizes = 0;
    }
    else
    {
        for (int dim = 0; dim < a->ndims; dim++)
        {
            if (a->shape[dim] != b->shape[dim])
            {
                equal_sizes = 0;
                break;
            }
        }
    }

    if (equal_sizes == 0)
    {

        fprintf(stderr, "[%s] shapes mismatch: shape_1 = [", __FUNCTION__);
        for (int i = 0; i < a->ndims; i++)
        {
            if (i != 0)
            {
                fprintf(stderr, ",");
            }
            fprintf(stderr, "%d", a->shape[i]);
        }
        fprintf(stderr, "], shape_2 = [");
        for (int i = 0; i < b->ndims; i++)
        {
            if (i != 0)
            {
                fprintf(stderr, ",");
            }
            fprintf(stderr, "%d", b->shape[i]);
        }
        fprintf(stderr, "]\n");
    }
    return equal_sizes;
}

int check_shapes_for_mul_mat(Matrix *m1, Matrix *m2, Matrix *result)
{
    /*
    check if sizes of given matrices are appropriate for dot product;
    return: 1 if appropriate, else 0

    dot product is only applicable for ndim = 2 matrices like 3x3 matrices
    */
    if (m1 == NULL || m2 == NULL || result == NULL)
    {
        return 0;
    }

    // only 2D
    if (m1->ndims != 2 || m2->ndims != 2 || result->ndims == 2)
    {
        return 0;
    }
    // === CHECK EQUAL INNER DIMS
    if (m1->shape[1] == m2->shape[0])
    {
        return 0;
    }
    if (result->shape[0] != m1->shape[0] || result->shape[1] != m2->shape[1])
    {
        return 0;
    }
    return 1;
}

void add_mat_to(Matrix *a, Matrix *b, Matrix *result)
{
    // === ERROR CHECK ===
    int eq = check_shapes_elementweise_op(a, b);
    int eq_result = check_shapes_elementweise_op(a, result);
    if (eq == 0 || eq_result == 0)
    {
        fprintf(stderr, "error in function [%s]\n", __FUNCTION__);
        exit(-1);
    }
    // === ERROR CHECK ===

    // === APPLY OPERATION ===
    int ndims = a->ndims;
    int *idx = calloc(ndims, sizeof(int)); // current indices

    for (int count = 0; count < a->size; count++)
    {
        // Compute current offsets using stride and index counter
        int offset_a = 0, offset_b = 0, offset_c = 0;
        for (int d = 0; d < ndims; d++)
        {
            offset_a += idx[d] * a->stride[d];
            offset_b += idx[d] * b->stride[d];
            offset_c += idx[d] * result->stride[d];
        }

        result->data[offset_c] = a->data[offset_a] + b->data[offset_b];

        // compute next index
        for (int d = ndims - 1; d >= 0; d--)
        {
            idx[d]++;
            if (idx[d] < a->shape[d])
                break;
            idx[d] = 0;
        }
    }
    free(idx);
    // === APPLY OPERATION ===
}

void sub_mat_to(Matrix *a, Matrix *b, Matrix *result)
{
    // === ERROR CHECK ===
    int eq = check_shapes_elementweise_op(a, b);
    int eq_result = check_shapes_elementweise_op(a, result);
    if (eq == 0 || eq_result == 0)
    {
        fprintf(stderr, "error in function [%s]\n", __FUNCTION__);
        exit(-1);
    }
    // === ERROR CHECK ===

    // === APPLY OPERATION ===
    int ndims = a->ndims;
    int *idx = calloc(ndims, sizeof(int)); // current indices

    for (int count = 0; count < a->size; count++)
    {
        // Compute current offsets using stride and index counter
        int offset_a = 0, offset_b = 0, offset_c = 0;
        for (int d = 0; d < ndims; d++)
        {
            offset_a += idx[d] * a->stride[d];
            offset_b += idx[d] * b->stride[d];
            offset_c += idx[d] * result->stride[d];
        }

        // subtract values
        result->data[offset_c] = a->data[offset_a] - b->data[offset_b];

        // compute next index
        for (int d = ndims - 1; d >= 0; d--)
        {
            idx[d]++;
            if (idx[d] < a->shape[d])
                break;
            idx[d] = 0;
        }
    }
    free(idx);
    // === APPLY OPERATION ===
}

void mul_mat_to(Matrix *a, Matrix *b, Matrix *result)
{
    // === CHECK ERROR ===
    int check_results = check_shapes_for_mul_mat(a, b, result);
    if (check_results == 0)
    {
        fprintf(stderr, "[%s] dot product not possible\n", __FUNCTION__);
        exit(-1);
    }
    // === CHECK ERROR ===

    for (int y = 0; y < a->shape[0]; y++)
    {
        for (int x = 0; x < a->shape[1]; x++)
        {
            double acc = 0;
            for (int i = 0; i < a->shape[1]; i++)
            {
                double a_value = a->data[y * a->stride[0] + i];
                double b_value = b->data[i * b->stride[0] + x];
                double r_value = a_value * b_value;
                acc += r_value;
            }
            result->data[y * result->stride[0] + x] = acc;
        }
    }
}

void e_div_mat_to(Matrix *a, Matrix *b, Matrix *result)
{
    // === ERROR CHECK ===
    int eq = check_shapes_elementweise_op(a, b);
    int eq_result = check_shapes_elementweise_op(a, result);
    if (eq == 0 || eq_result == 0)
    {
        fprintf(stderr, "error in function [%s]\n", __FUNCTION__);
        exit(-1);
    }
    // === ERROR CHECK ===

    // === APPLY OPERATION ===
    int ndims = a->ndims;
    int *idx = calloc(ndims, sizeof(int)); // current indices

    for (int count = 0; count < a->size; count++)
    {
        // Compute current offsets using stride and index counter
        int offset_a = 0, offset_b = 0, offset_c = 0;
        for (int d = 0; d < ndims; d++)
        {
            offset_a += idx[d] * a->stride[d];
            offset_b += idx[d] * b->stride[d];
            offset_c += idx[d] * result->stride[d];
        }

        assert(b->data[offset_b] != 0); // division by zero
        result->data[offset_c] = a->data[offset_a] / b->data[offset_b];

        // compute next index
        for (int d = ndims - 1; d >= 0; d--)
        {
            idx[d]++;
            if (idx[d] < a->shape[d])
                break;
            idx[d] = 0;
        }
    }
    free(idx);
    // === APPLY OPERATION ===
}

void e_mul_mat_to(Matrix *a, Matrix *b, Matrix *result)
{
    // === ERROR CHECK ===
    int eq = check_shapes_elementweise_op(a, b);
    int eq_result = check_shapes_elementweise_op(a, result);
    if (eq == 0 || eq_result == 0)
    {
        fprintf(stderr, "error in function [%s]\n", __FUNCTION__);
        exit(-1);
    }
    // === ERROR CHECK ===

    // === APPLY OPERATION ===
    int ndims = a->ndims;
    int *idx = calloc(ndims, sizeof(int)); // current indices

    for (int count = 0; count < a->size; count++)
    {
        // Compute current offsets using stride and index counter
        int offset_a = 0, offset_b = 0, offset_c = 0;
        for (int d = 0; d < ndims; d++)
        {
            offset_a += idx[d] * a->stride[d];
            offset_b += idx[d] * b->stride[d];
            offset_c += idx[d] * result->stride[d];
        }
        result->data[offset_c] = a->data[offset_a] * b->data[offset_b];

        // compute next index
        for (int d = ndims - 1; d >= 0; d--)
        {
            idx[d]++;
            if (idx[d] < a->shape[d])
                break;
            idx[d] = 0;
        }
    }
    free(idx);
    // === APPLY OPERATION ===
}

void e_pow_mat_to(Matrix *m, Matrix *result, double pow_value)
{
    // === ERROR CHECK ===
    int eq_result = check_shapes_elementweise_op(m, result);
    if (eq_result == 0)
    {
        fprintf(stderr, "error in function [%s]\n", __FUNCTION__);
        exit(-1);
    }
    // === ERROR CHECK ===

    // === APPLY OPERATION ===
    int ndims = m->ndims;
    int *idx = calloc(ndims, sizeof(int)); // current indices

    for (int count = 0; count < m->size; count++)
    {
        // Compute current offsets using stride and index counter
        int offset_m = 0, offset_r = 0;
        for (int d = 0; d < ndims; d++)
        {
            offset_m += idx[d] * m->stride[d];
            offset_r += idx[d] * result->stride[d];
        }
        // subtract values
        double m_value = m->data[offset_m];
        result->data[offset_r] = pow(m_value, pow_value);

        // compute next index
        for (int d = ndims - 1; d >= 0; d--)
        {
            idx[d]++;
            if (idx[d] < m->shape[d])
                break;
            idx[d] = 0;
        }
    }
    free(idx);
    // === APPLY OPERATION ===
}

void transpose_mat(Matrix *m, int dim1, int dim2)
{
    if (m == NULL)
    {
        fprintf(stderr, "input matrix is null\n");
        exit(-1);
    }
    if (dim1 < 0 || dim1 >= m->ndims)
    {
        fprintf(stderr, "[%s] dim1 = %d out of range\n", __FUNCTION__, dim1);
        exit(-1);
    }
    if (dim1 < 0 || dim1 >= m->ndims)
    {
        fprintf(stderr, "[%s] dim2 = %d out of range\n", __FUNCTION__, dim2);
        exit(-1);
    }
    // swap shape
    int temp_value = m->shape[dim1];
    m->shape[dim1] = m->shape[dim2];
    m->shape[dim2] = m->shape[temp_value];

    // swap stride
    temp_value = m->stride[dim1];
    m->stride[dim1] = m->stride[dim2];
    m->stride[dim2] = m->stride[temp_value];
}

void scale_mat_inplace(Matrix *m, double scaler)
{
    // === ERROR CHECK ===
    if (m == NULL)
    {
        fprintf(stderr, "[%s] m is null\n", __FUNCTION__);
        exit(-1);
    }
    // === ERROR CHECK ===

    // === APPLY OPERATION ===
    int ndims = m->ndims;
    int *idx = calloc(ndims, sizeof(int)); // current indices

    for (int count = 0; count < m->size; count++)
    {
        // Compute current offsets using stride and index counter
        int offset_m = 0;
        for (int d = 0; d < ndims; d++)
        {
            offset_m += idx[d] * m->stride[d];
        }
        // subtract values
        double m_value = m->data[offset_m];
        m->data[offset_m] = m_value * scaler;

        // compute next index
        for (int d = ndims - 1; d >= 0; d--)
        {
            idx[d]++;
            if (idx[d] < m->shape[d])
                break;
            idx[d] = 0;
        }
    }
    free(idx);
    // === APPLY OPERATION ===
}

double dot_mat(Matrix *a, Matrix *b)
{
    // === ERROR CHECK ===
    int eq = check_shapes_elementweise_op(a, b);
    if (eq == 0)
    {
        fprintf(stderr, "error in function [%s]\n", __FUNCTION__);
        exit(-1);
    }
    if (a->ndims != 1 || b->ndims != 1)
    {
        fprintf(stderr, "[%s] dot product is only applicable for 1D matrices\n", __FUNCTION__);
        exit(-1);
    }
    // === ERROR CHECK ===

    // === APPLY OPERATION ===
    int ndims = a->ndims;
    int *idx = calloc(ndims, sizeof(int)); // current indices
    double result = 0;

    for (int count = 0; count < a->size; count++)
    {
        // Compute current offsets using stride and index counter
        int offset_a = 0, offset_b = 0, offset_c = 0;
        for (int d = 0; d < ndims; d++)
        {
            offset_a += idx[d] * a->stride[d];
            offset_b += idx[d] * b->stride[d];
        }

        result += a->data[offset_a] * b->data[offset_b];

        // compute next index
        for (int d = ndims - 1; d >= 0; d--)
        {
            idx[d]++;
            if (idx[d] < a->shape[d])
                break;
            idx[d] = 0;
        }
    }
    free(idx);
    // === APPLY OPERATION ===
    return result;
}

Matrix *slice_mat(Matrix *m, int *slice_range, int slice_range_size)
{
    // === ERROR CHECK ===
    if (m == NULL)
    {
        fprintf(stderr, "[%s] m is null\n", __FUNCTION__);
        exit(-1);
    }
    if (slice_range_size != (m->ndims * 2))
    {

        fprintf(stderr, "[%s] slice range size != ndims * 2\n", __FUNCTION__);
        exit(-1);
    }
    // === RANGE TO SHAPE
    int *slice_shape = calloc(m->ndims, sizeof(int));
    for (int dim = 0; dim < m->ndims; dim++)
    {
        int range_index = dim * 2;
        int start = slice_range[range_index];
        int end = slice_range[range_index + 1];
        if (start < 0 || end > m->shape[dim])
        {
            fprintf(stderr, "range of dim %d is out of bounds\n", dim);
            exit(-1);
        }
        slice_shape[dim] = end - start;
    }
    // === RANGE TO SHAPE

    // === ERROR CHECK ===

    Matrix *sliced = new_view(slice_shape, m->ndims, m->stride);

    int offset = 0;
    for (int i = 0; i < m->ndims; i++)
    {
        offset += slice_range[i * 2] * m->stride[i];
    }

    sliced->data = m->data + (offset);

    return sliced;
}

Matrix *add_mat(Matrix *a, Matrix *b)
{
    Matrix *r = new_mat(a->shape, a->ndims);
    add_mat_to(a, b, r);
    return r;
}

Matrix *sub_mat(Matrix *a, Matrix *b)
{
    Matrix *r = new_mat(a->shape, a->ndims);
    sub_mat_to(a, b, r);
    return r;
}

void div_mat_by_value_to(Matrix *m, double value, Matrix *result)
{
    int eq = check_shapes_elementweise_op(m, result);
    if (eq == 0)
    {
        fprintf(stderr, "error in function [%s]\n", __FUNCTION__);
        exit(-1);
    }

    // === APPLY OPERATION ===
    int ndims = m->ndims;
    int *idx = calloc(ndims, sizeof(int)); // current indices

    for (int count = 0; count < m->size; count++)
    {
        // Compute current offsets using stride and index counter
        int offset_m = 0, offset_r = 0;
        for (int d = 0; d < ndims; d++)
        {
            offset_m += idx[d] * m->stride[d];
            offset_r += idx[d] * result->stride[d];
        }
        double m_value = m->data[offset_m];
        result->data[offset_r] = m_value / value;

        // compute next index
        for (int d = ndims - 1; d >= 0; d--)
        {
            idx[d]++;
            if (idx[d] < m->shape[d])
                break;
            idx[d] = 0;
        }
    }
    free(idx);
    // === APPLY OPERATION ===
}

Matrix *div_mat_by_value(Matrix *m, double value)
{
    Matrix *result = new_mat(m->shape, m->ndims);
    div_mat_by_value_to(m, value, result);
    return result;
}
void scale_mat_to(Matrix *m, double scaler, Matrix *result)
{
    // === ERROR CHECK ===
    int eq = check_shapes_elementweise_op(m, result);
    if (eq == 0)
    {
        fprintf(stderr, "[%s] m is null\n", __FUNCTION__);
        exit(-1);
    }
    // === ERROR CHECK ===

    // === APPLY OPERATION ===
    int ndims = m->ndims;
    int *idx = calloc(ndims, sizeof(int)); // current indices

    for (int count = 0; count < m->size; count++)
    {
        // Compute current offsets using stride and index counter
        int offset_m = 0, offset_r = 0;
        for (int d = 0; d < ndims; d++)
        {
            offset_m += idx[d] * m->stride[d];
            offset_r += idx[d] * result->stride[d];
        }
        // subtract values
        double m_value = m->data[offset_m];
        result->data[offset_r] = m_value * scaler;

        // compute next index
        for (int d = ndims - 1; d >= 0; d--)
        {
            idx[d]++;
            if (idx[d] < m->shape[d])
                break;
            idx[d] = 0;
        }
    }
    free(idx);
    // === APPLY OPERATION ===
}

double max(Matrix *m)
{
    if (m == NULL)
    {
        fprintf(stderr, "matrix is null\n");
        exit(-1);
    }
    double max = -INFINITY;

    int ndims = m->ndims;
    int *idx = calloc(ndims, sizeof(int)); // current indices

    for (int count = 0; count < m->size; count++)
    {
        // Compute current offsets using stride and index counter
        int offset_m = 0, offset_r = 0;
        for (int d = 0; d < ndims; d++)
        {
            offset_m += idx[d] * m->stride[d];
        }
        // subtract values
        double value = m->data[offset_m];
        if (value > max)
        {
            max = value;
        }

        // compute next index
        for (int d = ndims - 1; d >= 0; d--)
        {
            idx[d]++;
            if (idx[d] < m->shape[d])
                break;
            idx[d] = 0;
        }
    }
    free(idx);

    return max;
}
int argmax(Matrix *m)
{
    if (m == NULL)
    {
        fprintf(stderr, "[%s] matrix is null\n", __FUNCTION__);
        exit(-1);
    }
    if (m->ndims != 1 || m->ndims != 2)
    {
        fprintf(stderr, "[%s] argmax is only for 1D and 2D with first dim = 1 matrices applicable\n", __FUNCTION__);
        exit(-1);
    }

    double max = -INFINITY;
    int arg = -1;

    int ndims = m->ndims;
    int *idx = calloc(ndims, sizeof(int)); // current indices

    for (int count = 0; count < m->size; count++)
    {
        // Compute current offsets using stride and index counter
        int offset_m = 0, offset_r = 0;
        for (int d = 0; d < ndims; d++)
        {
            offset_m += idx[d] * m->stride[d];
        }
        // subtract values
        double value = m->data[offset_m];
        if (value > max)
        {
            max = value;
            arg = count;
        }

        // compute next index
        for (int d = ndims - 1; d >= 0; d--)
        {
            idx[d]++;
            if (idx[d] < m->shape[d])
                break;
            idx[d] = 0;
        }
    }
    return arg;
}

void print_mat_rec(Matrix *m, int *indices, int level)
{

    for (int i = 0; i < level; ++i)
    {
        printf("    ");
    }
    if (level == m->ndims - 1)
    {
        printf("[");
        for (int i = 0; i < m->shape[level]; ++i)
        {
            int flat_index = 0;
            for (int j = 0; j < m->ndims; j++)
            {
                flat_index += m->stride[j] * indices[j];
            }
            flat_index += i;
            printf("%.2f", m->data[flat_index]);
            if (i < m->shape[level] - 1)
                printf(", ");
        }
        printf("]");
    }
    else
    {
        printf("[\n");
        for (int i = 0; i < m->shape[level]; ++i)
        {
            indices[level] = i;
            print_mat_rec(m, indices, level + 1);
            if (i < m->shape[level] - 1)
                printf(", \n");
        }
        printf("\n");
        for (int i = 0; i < level; ++i)
        {
            printf("    ");
        }
        printf("]");
    }
    if (level == 0)
        printf("\n");
}
void print_mat(Matrix *m)
{
    printf("shape: [");
    for (int dim = 0; dim < m->ndims; ++dim)
    {
        if (dim != 0)
        {
            printf(", ");
        }
        printf("%d", m->shape[dim]);
    }
    printf("]\n");
    printf("stride: [");
    for (int dim = 0; dim < m->ndims; ++dim)
    {
        if (dim != 0)
        {
            printf(", ");
        }
        printf("%d", m->stride[dim]);
    }
    printf("]\n");
    int *indices = calloc(m->ndims, sizeof(int));
    print_mat_rec(m, indices, 0);
}

void copy_mat(Matrix *source, Matrix *target)
{
    assert(source != NULL);
    assert(source->data != NULL);
    assert(target != NULL);
    assert(target->data != NULL);
    for (int dim = 0; dim < source->ndims; ++dim)
    {
        assert(source->shape[dim] == target->shape[dim]);
    }
    int ndims = source->ndims;
    int *idx = calloc(ndims, sizeof(int)); // current indices

    for (int count = 0; count < source->size; count++)
    {
        // Compute current offsets using stride and index counter
        int offset_s = 0, offset_t = 0;
        for (int d = 0; d < ndims; d++)
        {
            offset_s += idx[d] * source->stride[d];
            offset_t += idx[d] * target->stride[d];
        }

        target->data[offset_t] = source->data[offset_s];

        // compute next index
        for (int d = ndims - 1; d >= 0; d--)
        {
            idx[d]++;
            if (idx[d] < source->shape[d])
                break;
            idx[d] = 0;
        }
    }
    free(idx);
}

void fill_mat_with(double value, Matrix *m)
{
    if (m == NULL)
    {
        printf("[%s] cannot fill null matrix\n", __FUNCTION__);
        return;
    }
    else if (m->data == NULL)
    {
        printf("[%s] cannot fill a matrix with null data\n", __FUNCTION__);
        return;
    }

    int ndims = m->ndims;
    int *idx = calloc(ndims, sizeof(int)); // current indices

    for (int count = 0; count < m->size; count++)
    {
        // Compute current offsets using stride and index counter
        int offset = 0;
        for (int d = 0; d < ndims; d++)
        {
            offset += idx[d] * m->stride[d];
        }

        m->data[offset] = value;

        // compute next index
        for (int d = ndims - 1; d >= 0; d--)
        {
            idx[d]++;
            if (idx[d] < m->shape[d])
                break;
            idx[d] = 0;
        }
    }
    free(idx);
}

void stepwise_fill_mat(double start, double step, Matrix *m)
{
    if (m == NULL)
    {
        printf("[%s] cannot fill null matrix\n", __FUNCTION__);
        return;
    }
    else if (m->data == NULL)
    {
        printf("[%s] cannot fill a matrix with null data\n", __FUNCTION__);
        return;
    }

    int ndims = m->ndims;
    int *idx = calloc(ndims, sizeof(int)); // current indices

    for (int count = 0; count < m->size; count++)
    {
        // Compute current offsets using stride and index counter
        int offset = 0;
        for (int d = 0; d < ndims; d++)
        {
            offset += idx[d] * m->stride[d];
        }

        m->data[offset] = start;
        start += step;

        // compute next index
        for (int d = ndims - 1; d >= 0; d--)
        {
            idx[d]++;
            if (idx[d] < m->shape[d])
                break;
            idx[d] = 0;
        }
    }
    free(idx);
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
    assert(m != 0);
    assert(m->data != 0);

    int ndims = m->ndims;
    int *idx = calloc(ndims, sizeof(int)); // current indices

    for (int count = 0; count < m->size; count++)
    {
        // Compute current offsets using stride and index counter
        int offset = 0;
        for (int d = 0; d < ndims; d++)
        {
            offset += idx[d] * m->stride[d];
        }

        double value = get_random_number();
        m->data[offset] = value;

        // compute next index
        for (int d = ndims - 1; d >= 0; d--)
        {
            idx[d]++;
            if (idx[d] < m->shape[d])
                break;
            idx[d] = 0;
        }
    }
    free(idx);
}

Matrix *random_mat(int *shape, int ndims)
{
    Matrix *m = new_mat(shape, ndims);
    random_fill_mat(m);
    return m;
}