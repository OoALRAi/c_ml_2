#include <stdio.h>
#include <stdlib.h>
int main()
{
    char *str = "123.1000,0,21";
    double value = atof(str);
    printf("%f\n", value);
    value = atof(str);
    printf("%f\n", value);
    return 0;
}