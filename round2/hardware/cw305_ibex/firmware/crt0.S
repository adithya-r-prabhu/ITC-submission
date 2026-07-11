    .section .text.init
    .global _start
_start:
    li   sp, 0x00103FF0
    li   gp, 0x00101000
    call main
hang:
    j    hang
