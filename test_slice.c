#include "matrix.h"
int main(void)
{
    int shape[] = {3, 10, 10};
    Matrix *img = new_mat(shape, 3);
    stepwise_fill_mat(0, 1, img);
    printf("input image: \n");
    print_mat(img);

    int slice_range[] = {0, 3, 0, 3, 0, 3};
    Matrix *img_slice = slice_mat(img, slice_range, 3 * 2);
    print_mat(img_slice);

    return 0;
}