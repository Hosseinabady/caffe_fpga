#ifndef __FPGA_ACCEL_CNN_LAYERS_HPP__
#define __FPGA_ACCEL_CNN_LAYERS_HPP__
#include "caffe/fpga_accel_basic.hpp"
void fpga_conv_layer(
	data_t *data_input_0,
	data_t *data_input_1,
	data_t *data_input_2,
	data_t *data_input_3,


	data_t* data_output_in,
	data_t* data_output_out,
	weight_t*  w,
	weight_t*  w_bias,

	param_t    n,

	param_t          param_input_h,
	param_t          param_input_w,
	param_t          param_input_c,
	param_t          param_input_s,

	param_t          param_output_h,
	param_t          param_output_w,
	param_t          param_output_c,
	param_t          param_output_s,

	param_t          output_p,

	param_t          param_kernel_h,
	param_t          param_kernel_w
);
/*
#include "caffe/fpga_accel_basic.hpp"

void fpga_accel_cnn(
	data_t        data_input[DATA_HEIGHT_MAX*DATA_WIDTH_MAX],
	data_t        data_output[DATA_HEIGHT_MAX*DATA_WIDTH_MAX],
	weight_t      w[NO_OUT_CHANNEL_MAX*NO_IN_CHANNEL_MAX * 3 * 3],
	weight_t      w_bias[DATA_HEIGHT_MAX],
	
	param_t       n,

	param_t       param_input_h,
	param_t       param_input_w,
	param_t       param_input_c,
	param_t       param_input_s,

	param_t       param_output_h,
	param_t       param_output_w,
	param_t       param_output_c,
	param_t       param_output_s,

	param_t       param_output_p,

	param_t       param_kernel_h,
	param_t       param_kernel_w
);
*/

#endif //__FPGA_ACCEL_CNN_LAYERS_HPP__
