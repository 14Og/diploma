ENTRY(Reset_Handler)

MEMORY {
  FLASH (rx) : ORIGIN = 0x08000000, LENGTH = 512K
  RAM   (rwx): ORIGIN = 0x20000000, LENGTH = 64K
}

SECTIONS {
  .isr_vector : {
    KEEP(*(.isr_vector))
  } > FLASH

  .text : {
    *(.text*)
    *(.rodata*)
  } > FLASH

  .data : AT(_etext) {
    _sdata = .;
    *(.data*)
    _edata = .;
  } > RAM

  .bss : {
    _sbss = .;
    *(.bss*)
    *(COMMON)
    _ebss = .;
  } > RAM

  _etext = LOADADDR(.data);
  _sidata = LOADADDR(.data);
}

PROVIDE(__exidx_start = 0);
PROVIDE(__exidx_end = 0);
_end = .;
PROVIDE(end = .);