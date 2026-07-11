OUTPUT_ARCH(riscv)
ENTRY(_start)
MEMORY { RAM (rwxai) : ORIGIN = 0x00100000, LENGTH = 16K }
SECTIONS {
  . = 0x00100000;
  .text : { *(.text.init) *(.text*) } > RAM
  .rodata : { *(.rodata*) *(.srodata*) } > RAM
  .data : { *(.data*) *(.sdata*) } > RAM
  .bss : { *(.bss*) *(.sbss*) *(COMMON) } > RAM
  . = ALIGN(4);
  _end = .;
}
