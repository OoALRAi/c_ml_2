#include <stdio.h>
#include "nn.h"

int main(void)
{
    Conv2d *conv = create_conv2d(5, 1, 5, 1, 0);
    print_conv2d(conv);
    return 0;
}