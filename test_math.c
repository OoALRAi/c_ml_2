
#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include <assert.h>

void test_addition()
{
    // simple addition
    int shape1[] = {2, 2};
    Matrix *m1 = new_mat(shape1, 2);
    fill_mat_with(3, m1);
    Matrix *m2 = new_mat(shape1, 2);
    fill_mat_with(2, m2);
    Matrix *r = add_mat(m1, m2);
    Matrix *gt_r = new_mat(shape1, 2);
    fill_mat_with(5, gt_r);
    assert(equals(r, gt_r) == 1);
    free_mat(m1);
    free_mat(m2);
    free_mat(r);
    free_mat(gt_r);

    // add different shapes
    int shape2[] = {2, 2, 3};
    int shape3[] = {2, 3};
    m1 = new_mat(shape2, 3);
    stepwise_fill_mat(2, 3, m1);
    m2 = new_mat(shape3, 2);
    stepwise_fill_mat(1, 1, m2);
    r = add_mat(m1, m2);
}
void test_subtraction()
{
    // simple sub
    int shape1[] = {2, 2};
    Matrix *m1 = new_mat(shape1, 2);
    fill_mat_with(9, m1);
    Matrix *m2 = new_mat(shape1, 2);
    fill_mat_with(2, m2);
    Matrix *r = sub_mat(m1, m2);
    Matrix *gt_r = new_mat(shape1, 2);
    fill_mat_with(7, gt_r);
    assert(equals(r, gt_r) == 1);
    free_mat(m1);
    free_mat(m2);
    free_mat(r);
    free_mat(gt_r);

    // sub different shapes
    int shape2[] = {2, 2, 3};
    int shape3[] = {2, 3};
    m1 = new_mat(shape2, 3);
    stepwise_fill_mat(2, 3, m1);
    m2 = new_mat(shape3, 2);
    stepwise_fill_mat(1, 1, m2);
    r = sub_mat(m1, m2);
}

void test_transpose()
{
    int shape[] = {1, 5};
    Matrix *m = new_mat(shape, 2);
    stepwise_fill_mat(1, 1, m);
    Matrix *r = transpose_mat(m, 0, 1);
    print_mat(m);
    print_mat(r);
}

void test_dot()
{
    // === 1D dot product ===
    int shape1[] = {4};
    int shape2[] = {4};
    Matrix *a = new_mat(shape1, 1);
    Matrix *b = new_mat(shape2, 1);
    fill_mat_with(1, a);
    fill_mat_with(7, b);
    double result = dot_mat(a, b);
    assert(result == 28);
    printf("test1: succeed\n");
    free_mat(a);
    free_mat(b);
    // === 1D dot product ===

    // === 2D dot product ===
    int shape3[] = {2, 3};
    int shape4[] = {2, 3};
    a = new_mat(shape3, 2);
    b = new_mat(shape4, 2);
    fill_mat_with(2, a);
    fill_mat_with(3, b);
    result = dot_mat(a, b);
    assert(result == 2 * 3 * 6);
    printf("test2: succeed\n");
    free_mat(a);
    free_mat(b);

    // ===

    int shape5[] = {3, 10, 10};
    a = new_mat(shape5, 3);
    stepwise_fill_mat(1, 1, a);

    int shape6[] = {3, 3};
    b = new_mat(shape6, 2);
    fill_mat_with(3, b);

    int slice_range[] = {0, 1, 1, 4, 2, 5};
    Matrix *image_slice = slice_mat(a, slice_range, 6);
    squeeze_first_dim(image_slice);
    result = dot_mat(b, image_slice);
    assert(result == 648);
    printf("test3: succeed\n");
    free_mat(a);
    free_mat(b);
    free_mat(image_slice);
    // === 2D dot product ===
}
void test_mat_mul()
{
    int shape1[] = {1, 3};
    int shape2[] = {3, 2};
    Matrix *m1 = new_mat(shape1, 2);
    Matrix *m2 = new_mat(shape2, 2);
    fill_mat_with(1, m1);
    fill_mat_with(2, m2);
    Matrix *r = mul_mat(m1, m2);
    print_mat(r);
}

void test_conv_mul_mat()
{
    int shape[] = {3,     // 3 channels image
                   5, 5}; // height x wdith = 4 x 4 image and zero-padding of 1

    int kernel_shape[] = {3,     // channels
                          3, 3}; // 3 x 3 kernel filter

    Matrix *image = new_mat(shape, 3);
    Matrix *kernel = new_mat(kernel_shape, 3);
}

int main(void)
{
    // test_addition();
    // test_subtraction();
    // printf("[TEST DOT PRODUCT]\n");
    // test_dot();
    test_transpose();
    // test_mat_mul();
    // test_conv_mul_mat();
    return 0;
}