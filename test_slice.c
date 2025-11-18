#include "matrix.h"
int main(void)
{

    int shape[3] = {4, 4, 4};
    Matrix *m = new_mat(shape, 3);
    stepwise_fill_mat(10, 2, m);
    int slice_range[6] = {1, 3, 1, 3, 1, 3};
    Matrix *view = slice_mat(m, slice_range, 6);
    print_mat(m);
    print_mat(view);

    return 0;
}