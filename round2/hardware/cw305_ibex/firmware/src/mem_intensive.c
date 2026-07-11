#include "simple_system_common.h"

#define BLOCK 16
static volatile unsigned int block[BLOCK];
static volatile unsigned int acc_sink;

int main(void) {
    pcount_enable(0);
    pcount_reset();
    pcount_enable(1);

    unsigned int acc = 0;
    for (int outer = 0; outer < 60; outer++) {
        for (int i = 0; i < BLOCK; i++)
            block[i] = (unsigned int)(i + outer);
        for (int i = 0; i < BLOCK; i++)
            acc += block[i];
    }

    acc_sink = acc;
    puthex(acc_sink); putchar('\n');

    pcount_enable(0);
    sim_halt();
    return 0;
}
