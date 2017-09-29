#ifndef FPGA_H__
#define FPGA_H__
#include <cstddef>

void *fpga_malloc(size_t size);
void  fpga_free(void* ptr);



#endif //FPGA_H__

