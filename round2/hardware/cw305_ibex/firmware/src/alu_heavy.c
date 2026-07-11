#include "simple_system_common.h"

static volatile unsigned int result;

int main(void) {
    pcount_enable(0);
    pcount_reset();
    pcount_enable(1);

    unsigned int a = 0x12345678u;
    unsigned int b = 0xDEADBEEFu;
    unsigned int c = 0u;
    unsigned int d = 0u;

    for (int i = 0; i < 200; i++) {
        c = a ^ b;
        d = a + b;
        a = (b << 1) | (b >> 31);
        b = c + d;
        b ^= (unsigned int)i;
    }

    result = b;
    puthex(result);
    putchar('\n');

    pcount_enable(0);
    sim_halt();
    return 0;
}
