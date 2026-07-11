#ifndef SIMPLE_SYSTEM_COMMON_STUB_H
#define SIMPLE_SYSTEM_COMMON_STUB_H
#include <stdint.h>

#define MMIO_RESULT 0x10000000u
#define DEV_WRITE(addr, val) (*((volatile uint32_t *)(addr)) = (val))
#define DEV_READ(addr, val)  (*((volatile uint32_t *)(addr)))

static inline int putchar(int c) { (void)c; return c; }
static inline int puts(const char *s) { (void)s; return 0; }
static inline void puthex(uint32_t h) { DEV_WRITE(MMIO_RESULT, h); }
static inline void sim_halt(void) { }
static inline void pcount_enable(int e) { (void)e; }
static inline void pcount_reset(void) { }
static inline void timer_enable(uint64_t t) { (void)t; }
static inline void timecmp_update(uint64_t t) { (void)t; }
static inline void timer_disable(void) { }
static inline uint64_t get_elapsed_time(void) { return 0; }
static inline void icache_enable(int e) { (void)e; }
#endif
