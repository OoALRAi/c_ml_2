
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
void test_dot()
{
    int shape1[] = {4};
    Matrix *m1 = new_mat(shape1, 1);
    fill_mat_with(1, m1);

    Matrix *m2 = new_mat(shape1, 1);
    fill_mat_with(2, m2);

    double r = dot_mat(m1, m2);
    printf("dot result: %.2f\n", r);
    assert(r == 2 * 4);
}
void test_mat_mul() {}
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
    test_dot();
    test_mat_mul();
    test_conv_mul_mat();
    return 0;
}