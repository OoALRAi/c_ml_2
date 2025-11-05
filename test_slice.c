#include "matrix.h"
int main(void)
{
    Matrix *m = new_mat(4, 4);

    stepwise_fill_mat(10, 2, m);
    Matrix *view = slice_mat(m, 1, 2, 0, 4);
    print_mat(m);
    print_mat(view);

    return 0;
}