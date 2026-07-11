#include "simple_system_common.h"

static volatile unsigned int odd_sink;
static volatile unsigned int mod4_sink;
static volatile unsigned int exit_sink;

int main(void) {
    pcount_enable(0);
    pcount_reset();
    pcount_enable(1);

    unsigned int odd = 0, mod4 = 0, exits = 0;
    for (int outer = 0; outer < 40; outer++) {
        int counter = 0;
        const int limit = 100;
        while (counter < limit) {
            counter++;
            if ((counter & 1) != 0) {
                odd++;
            } else {
                if ((counter & 3) == 0) {
                    mod4++;
                }
            }
            if (counter >= limit) {
                exits++;
            }
        }
    }

    odd_sink = odd; mod4_sink = mod4; exit_sink = exits;
    puthex(odd_sink); putchar('\n');
    puthex(mod4_sink); putchar('\n');

    pcount_enable(0);
    sim_halt();
    return 0;
}
