#ifndef SOURCES_STARTUP_CPP_
#define SOURCES_STARTUP_CPP_

#include <stdint.h>
#include <stddef.h>

extern uint32_t _sidata, _sdata, _edata;
extern uint32_t _sbss, _ebss;

int main(void);

extern "C" {
    
[[noreturn]] void Reset_Handler(void);
void Default_Handler(void);
void *memcpy(void *dest, const void *src, size_t n);

__attribute__((section(".isr_vector"))) const void *vector_table[] = {
	(void *)(0x20010000),
	(void *)Reset_Handler,
};
}

void Default_Handler(void)
{
	while (1);
}

[[noreturn]] void Reset_Handler(void)
{
    uint32_t* src = &_sidata;
    uint32_t* dst = &_sdata;
    while (dst < &_edata) *dst++ = *src++;

    dst = &_sbss;
    while (dst < &_ebss) *dst++ = 0;

    main();
    while (1);
}

void *memcpy(void *dest, const void *src, size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        ((char*)dest)[i] = ((char*)src)[i];
    }
    return dest;
}

#endif /* SOURCES_STARTUP_CPP_ */
