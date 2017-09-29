#include "caffe/fpga.hpp"
#include "caffe/fpga_accel_driver.hpp"
#include <stdlib.h>
#include <iostream>

void *fpga_malloc(size_t size) {
#ifdef WIN32
	void *ptr = malloc(size);
	std::cout << "==================from fpga_malloc on WIN32 size = " << size << " at address " << ptr << std::endl;
#else
	void *ptr = fpgacl_malloc(size);
	std::cout << "==================from fpga_malloc on Zynq driver size = " << size << " at address " << ptr << std::endl;
#endif
	return ptr;
}

void  fpga_free(void* ptr) {
#ifdef WIN32
	free(ptr);
	std::cout << "==================from fpga_free on WIN32 addrss  " << ptr << std::endl;
#else
	fpgacl_free(ptr);
	std::cout << "==================from fpga_free on Zynq driver addrss  " << ptr << std::endl;
#endif
}
