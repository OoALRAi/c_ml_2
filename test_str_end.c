#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int main(void)
{
    char *str = calloc(2, sizeof(char));
    str[0] = '1';
    str[1] = '2';

    int soll_len = 2;
    int counter = 0;
    char *current_char = str;
    while (current_char[0] != '\0')
    {
        counter++;
        current_char++;
    }
    assert(soll_len == counter);
    printf("soll len = %d, actual len = %d\n", soll_len, counter);

    return 0;
}