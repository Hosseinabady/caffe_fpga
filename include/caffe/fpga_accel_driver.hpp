#ifdef WIN32

#else

#ifndef __ACCEL_H__
#define __ACCEL_H__
#include "caffe/fpga_accel_basic.hpp"
#include "caffe/xcnn_accel_hw.h"
#include <unistd.h>

#define ACCELERATOR                                       0x43C00000
typedef unsigned long long int u64;
typedef unsigned long u32;
typedef union {
	float f;
	u32   u;
}u32_float_conversion;


typedef union {
	int   i;
	u32   u;
}u32_int_uint_conversion;

//#define ITERATION_NUM 1

void accel_prologue();
void accel_epilogue();
void  accel_ready();
//void  accel_hp0_write(	u32 image_in_address, u32 image_out_address, u32 image_hight, u32 image_width);
typedef float data_t;
void  accel_write(
				data_t  	*data_input_0,
				data_t  	*data_input_1,
				data_t  	*data_input_2,
				data_t  	*data_input_3,
				data_t      *data_output_in,
				data_t      *data_output_out,
				data_t      *weights,
				data_t      *weights_bias,

				u32          n,

				u32          param_input_h,
				u32          param_input_w,
				u32          param_input_c,
				u32          param_input_s,

				u32          param_output_h,
				u32          param_output_w,
				u32          param_output_c,
				u32          param_output_s,

				u32          output_p,

				u32          param_kernel_h,
				u32          param_kernel_w
		);

void  accel_start( );
void  accel_wait();
double getTimestamp();


void* fpgacl_malloc(unsigned long int size);
void* fpgacl_cacheable_malloc(unsigned long int size);
void* fpgacl_noncacheable_malloc(unsigned long int size);
void fpgacl_free(void* user_var);

#endif //__ACCEL_H__

#endif// WIN32
