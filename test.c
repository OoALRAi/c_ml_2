#include <stdio.h>
#include "matrix.h"
#include "nn.h"

int main()
{

    Matrix *m = new_mat(5, 1);
    Matrix *r = new_mat(5, 1);

    set_element_at(m, 0, 0, 1.3);
    set_element_at(m, 0, 1, 5.1);
    set_element_at(m, 0, 2, 2.2);
    set_element_at(m, 0, 3, 0.7);
    set_element_at(m, 0, 4, 1.1);
    softmax(m, r);
    print_mat(m);
    print_mat(r);

    return 0;
}