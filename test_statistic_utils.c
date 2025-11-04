#include "statistic_utils.h"
#include "matrix.h"
#include <stdio.h>

int main()
{
    Matrix *cm = create_confision_matrix(3);
    add_prediction(cm, 1, 0);
    add_prediction(cm, 1, 0);
    add_prediction(cm, 1, 2);
    add_prediction(cm, 0, 1);
    add_prediction(cm, 2, 2);
    print_confusion_mat(cm);

    // for (int cls = 0; cls < 3; cls++)
    // {
    //     int tp = get_TP_of(cm, cls);
    //     printf("TP(%d) = %d\n", cls, tp);
    // }
    // printf("\n");
    // for (int cls = 0; cls < 3; cls++)
    // {
    //     int fp = get_FP_of(cm, cls);
    //     printf("FP(%d) = %d\n", cls, fp);
    // }
    // printf("\n");
    // for (int cls = 0; cls < 3; cls++)
    // {
    //     int fn = get_FN_of(cm, cls);
    //     printf("FN(%d) = %d\n", cls, fn);
    // }
    print_stats(cm);
    return 0;
}

/*
    predicted as
gt  |0  1  2|
    +---+---+
0   |0  1  0|
1   |2  0  1|
2   |0  0  1|

TP(0) = 0
TP(1) = 0
TP(2) = 1

FP(0) = 2
FP(1) = 1
FP(2) = 0

FN(0) = 1
FN(1) = 3
FN(2) = 0
*/