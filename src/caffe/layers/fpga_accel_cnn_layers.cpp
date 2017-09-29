#ifdef WIN32

#include "caffe/fpga_accel_basic.hpp"
#include "vivado_hls/hls_stream.h"
#include "caffe/layers/fpga_hls_video_mem.hpp"

const int _input_h_ = 360;
const int _input_w_ = 480;
const int _input_c_ = 5;
const int _input_p_ = 2;
const int _input_h_p_ = _input_h_ + _input_p_;
const int _input_w_p_ = _input_w_ + _input_p_;


const int _output_c_ = 7;
const int _output_s_ = 2;
const int _output_h_ = _input_h_ / _output_s_;
const int _output_w_ = _input_w_ / _output_s_;

const int _kernel_h_ = 3;
const int _kernel_w_ = 3;



void read_weight(
	weight_t*    w,
	weight_t     weights_local[NO_IN_CHANNEL_MAX][KERNEL_HEIGHT_MAX][KERNEL_WIDTH_MAX],
	param_t      param_input_h,
	param_t      param_input_w,
	param_t      param_input_c,
	param_t      param_input_s,

	param_t      param_output_h,
	param_t      param_output_w,
	param_t      param_output_c,
	param_t      param_output_s,

	param_t      param_kernel_h,
	param_t      param_kernel_w,
	param_t      ch_out,
	param_t      ch_in)
{
	int ch_in_valid = (param_input_c - (ch_in + 4))< 0 ? (param_input_c - ch_in) : 4;
	for (param_t ch_in_index = 0; ch_in_index < ch_in_valid; ch_in_index++) {
#pragma HLS LOOP_TRIPCOUNT min=_input_c_ max=_input_c_ avg=_input_c_
		for (param_t row_krn = 0; row_krn < param_kernel_h; row_krn++) {
#pragma HLS LOOP_TRIPCOUNT min=_kernel_h_ max=_kernel_h_ avg=_kernel_h_
			for (param_t col_krn = 0; col_krn < param_kernel_w; col_krn++) {
#pragma HLS LOOP_TRIPCOUNT min=_kernel_w_ max=_kernel_w_ avg=_kernel_w_
#pragma HLS PIPELINE
				int index_tmp = ch_out*(param_kernel_h*param_kernel_w*param_input_c) + ch_in*(param_kernel_h*param_kernel_w) + ch_in_index*(param_kernel_h*param_kernel_w) + row_krn*param_kernel_w + col_krn;
				weight_t w_tmp = w[index_tmp];
				weights_local[ch_in_index][row_krn][col_krn] = w_tmp;
			}
		}
	}
}


void conv_layer_core(
	data_t        *data_input_0,
	data_t        *data_input_1,
	data_t        *data_input_2,
	data_t        *data_input_3,


	data_t        *data_output_in,
	data_t        *data_output_out,
	weight_t       weights_local[NO_IN_CHANNEL_MAX][KERNEL_HEIGHT_MAX][KERNEL_WIDTH_MAX],
	weight_t       weights_bias_local[NO_OUT_CHANNEL_MAX],

	param_t          n,

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
	param_t          param_kernel_w,
	param_t          ch_out,
	param_t          ch_in)
{

#pragma HLS DATAFLOW
	hls::stream<data_t> input_data_stream_0;
	hls::stream<data_t> input_data_stream_1;
	hls::stream<data_t> input_data_stream_2;
	hls::stream<data_t> input_data_stream_3;

	hls::LineBuffer<KERNEL_HEIGHT_MAX - 1, DATA_WIDTH_MAX + (KERNEL_WIDTH_MAX / 2) * 2, data_t> input_buffer_0;
	hls::LineBuffer<KERNEL_HEIGHT_MAX - 1, DATA_WIDTH_MAX + (KERNEL_WIDTH_MAX / 2) * 2, data_t> input_buffer_1;
	hls::LineBuffer<KERNEL_HEIGHT_MAX - 1, DATA_WIDTH_MAX + (KERNEL_WIDTH_MAX / 2) * 2, data_t> input_buffer_2;
	hls::LineBuffer<KERNEL_HEIGHT_MAX - 1, DATA_WIDTH_MAX + (KERNEL_WIDTH_MAX / 2) * 2, data_t> input_buffer_3;

	hls::Window<KERNEL_HEIGHT_MAX, KERNEL_WIDTH_MAX, data_t>                            window_buffer_0;
	hls::Window<KERNEL_HEIGHT_MAX, KERNEL_WIDTH_MAX, data_t>                            window_buffer_1;
	hls::Window<KERNEL_HEIGHT_MAX, KERNEL_WIDTH_MAX, data_t>                            window_buffer_2;
	hls::Window<KERNEL_HEIGHT_MAX, KERNEL_WIDTH_MAX, data_t>                            window_buffer_3;

	u32 mask_p = param_kernel_h / 2;

	hls::stream<hls::Window<KERNEL_HEIGHT_MAX, KERNEL_WIDTH_MAX, data_t> > window_buffer_stream_0;
	hls::stream<hls::Window<KERNEL_HEIGHT_MAX, KERNEL_WIDTH_MAX, data_t> > window_buffer_stream_1;
	hls::stream<hls::Window<KERNEL_HEIGHT_MAX, KERNEL_WIDTH_MAX, data_t> > window_buffer_stream_2;
	hls::stream<hls::Window<KERNEL_HEIGHT_MAX, KERNEL_WIDTH_MAX, data_t> > window_buffer_stream_3;

	hls::Window<KERNEL_HEIGHT_MAX, KERNEL_WIDTH_MAX, data_t>  window_buffer_tmp_0;
	hls::Window<KERNEL_HEIGHT_MAX, KERNEL_WIDTH_MAX, data_t>  window_buffer_tmp_1;
	hls::Window<KERNEL_HEIGHT_MAX, KERNEL_WIDTH_MAX, data_t>  window_buffer_tmp_2;
	hls::Window<KERNEL_HEIGHT_MAX, KERNEL_WIDTH_MAX, data_t>  window_buffer_tmp_3;


	hls::stream<data_t> data_output_out_stream;
	int no_in_channel_valid = (param_input_c - (ch_in + 4))< 0 ? (param_input_c - ch_in) : 4;

read_actor:
	for (u32 row_in = 0; row_in < param_input_h; row_in++) {
#pragma HLS LOOP_TRIPCOUNT min=_output_h_ max=_output_h_ avg=_output_h_
		for (u32 col_in = 0; col_in < param_input_w; col_in++) {
#pragma HLS LOOP_TRIPCOUNT min=_output_w_ max=_output_w_ avg=_output_w_
#pragma HLS PIPELINE


			u32 index_0 = n*param_input_h *param_input_w *param_input_c + ch_in*param_input_h*param_input_w + 0 * param_input_h*param_input_w + row_in*param_input_w + col_in;;
			u32 index_1 = n*param_input_h *param_input_w *param_input_c + ch_in*param_input_h*param_input_w + 0 * param_input_h*param_input_w + row_in*param_input_w + col_in;;
			u32 index_2 = n*param_input_h *param_input_w *param_input_c + ch_in*param_input_h*param_input_w + 0 * param_input_h*param_input_w + row_in*param_input_w + col_in;;
			u32 index_3 = n*param_input_h *param_input_w *param_input_c + ch_in*param_input_h*param_input_w + 0 * param_input_h*param_input_w + row_in*param_input_w + col_in;;


			if (no_in_channel_valid > 1)
				index_1 += 1 * param_input_h*param_input_w;
			if (no_in_channel_valid > 2)
				index_2 += 2 * param_input_h*param_input_w;
			if (no_in_channel_valid > 3)
				index_3 += 3 * param_input_h*param_input_w;

			data_t data_0 = data_input_0[index_0];
			data_t data_1 = data_input_1[index_1];
			data_t data_2 = data_input_2[index_2];
			data_t data_3 = data_input_3[index_3];

			input_data_stream_0 << data_0;
			input_data_stream_1 << data_1;
			input_data_stream_2 << data_2;
			input_data_stream_3 << data_3;

		}
	}


window_slide_actor:
	for (param_t row_in = 0; row_in < param_input_h + (output_p) * 2; row_in++) {
#pragma HLS LOOP_TRIPCOUNT min=_input_h_p_ max=_input_h_p_ avg=_input_h_p_
		for (param_t col_in = 0; col_in < param_input_w + (output_p) * 2; col_in++) {
#pragma HLS LOOP_TRIPCOUNT min=_input_w_p_ max=_input_w_p_ avg=_input_w_p_
#pragma HLS PIPELINE
			data_t  d_in_0;
			data_t  d_in_1;
			data_t  d_in_2;
			data_t  d_in_3;

			if ((row_in >= (output_p) && row_in < param_input_h + (output_p)) && (col_in >= (output_p) && col_in < param_input_w + (output_p))) {
				d_in_0 = input_data_stream_0.read();
				d_in_1 = input_data_stream_1.read();
				d_in_2 = input_data_stream_2.read();
				d_in_3 = input_data_stream_3.read();
			}
			else {
				d_in_0 = 0;  //zero padding value
				d_in_1 = 0;  //zero padding value
				d_in_2 = 0;  //zero padding value
				d_in_3 = 0;  //zero padding value
			}


			data_t  d_tmp_0;
			data_t  d_tmp_1;
			data_t  d_tmp_2;
			data_t  d_tmp_3;
			if (row_in == 0) {
				d_tmp_0 = 0;
				d_tmp_1 = 0;
				d_tmp_2 = 0;
				d_tmp_3 = 0;
			}
			else {
				d_tmp_0 = input_buffer_0.getval(0, col_in);
				d_tmp_1 = input_buffer_1.getval(0, col_in);
				d_tmp_2 = input_buffer_2.getval(0, col_in);
				d_tmp_3 = input_buffer_3.getval(0, col_in);
			}

			//=====================Update Line buffer=
			input_buffer_0.shift_pixels_up(col_in);
			input_buffer_0(param_kernel_h - 2, col_in) = d_in_0;

			input_buffer_1.shift_pixels_up(col_in);
			input_buffer_1(param_kernel_h - 2, col_in) = d_in_1;

			input_buffer_2.shift_pixels_up(col_in);
			input_buffer_2(param_kernel_h - 2, col_in) = d_in_2;

			input_buffer_3.shift_pixels_up(col_in);
			input_buffer_3(param_kernel_h - 2, col_in) = d_in_3;

			//========================================


			//=====================Update window data=
			window_buffer_0.shift_left();
			window_buffer_1.shift_left();
			window_buffer_2.shift_left();
			window_buffer_3.shift_left();

			window_buffer_0.insert(d_tmp_0, 0, param_kernel_w - 1);
			window_buffer_1.insert(d_tmp_1, 0, param_kernel_w - 1);
			window_buffer_2.insert(d_tmp_2, 0, param_kernel_w - 1);
			window_buffer_3.insert(d_tmp_3, 0, param_kernel_w - 1);

			for (param_t row_wndw = 1; row_wndw < KERNEL_HEIGHT_MAX - 1; row_wndw++) {
				if (row_wndw < param_kernel_w - 1) {
					window_buffer_0.insert(input_buffer_0.getval(row_wndw - 1, col_in), row_wndw, param_kernel_w - 1);
					window_buffer_1.insert(input_buffer_1.getval(row_wndw - 1, col_in), row_wndw, param_kernel_w - 1);
					window_buffer_2.insert(input_buffer_2.getval(row_wndw - 1, col_in), row_wndw, param_kernel_w - 1);
					window_buffer_3.insert(input_buffer_3.getval(row_wndw - 1, col_in), row_wndw, param_kernel_w - 1);
				}
			}

			window_buffer_0.insert(d_in_0, param_kernel_w - 1, param_kernel_w - 1);
			window_buffer_1.insert(d_in_1, param_kernel_w - 1, param_kernel_w - 1);
			window_buffer_2.insert(d_in_2, param_kernel_w - 1, param_kernel_w - 1);
			window_buffer_3.insert(d_in_3, param_kernel_w - 1, param_kernel_w - 1);
			//========================================

			int row_in_update = row_in - (mask_p * 2);
			int col_in_update = col_in - (mask_p * 2);
			if ((row_in_update >= 0) && row_in_update%param_output_s == 0 &&
				col_in_update >= 0 && col_in_update%param_output_s == 0)
			{
				window_buffer_stream_0 << window_buffer_0;
				window_buffer_stream_1 << window_buffer_1;
				window_buffer_stream_2 << window_buffer_2;
				window_buffer_stream_3 << window_buffer_3;
			}
		}
	}



filter_actor:
	for (param_t row_out = 0; row_out < param_output_h; row_out++) {
#pragma HLS LOOP_TRIPCOUNT min=_output_h_ max=_output_h_ avg=_output_h_
		for (param_t col_out = 0; col_out < param_output_w; col_out++) {
#pragma HLS LOOP_TRIPCOUNT min=_output_w_ max=_output_w_ avg=_output_w_
#pragma HLS PIPELINE
			window_buffer_tmp_0 = window_buffer_stream_0.read();
			window_buffer_tmp_1 = window_buffer_stream_1.read();
			window_buffer_tmp_2 = window_buffer_stream_2.read();
			window_buffer_tmp_3 = window_buffer_stream_3.read();

			data_t d_out = 0;
			for (param_t row_kernel = 0; row_kernel < KERNEL_HEIGHT_MAX; row_kernel++) {
				for (param_t col_kernel = 0; col_kernel < KERNEL_WIDTH_MAX; col_kernel++) {
					if (row_kernel < param_kernel_h  && col_kernel < param_kernel_w) {
						data_t w_0 = weights_local[0][row_kernel][col_kernel];
						data_t w_1 = weights_local[1][row_kernel][col_kernel];
						data_t w_2 = weights_local[2][row_kernel][col_kernel];
						data_t w_3 = weights_local[3][row_kernel][col_kernel];


						data_t d_0 = window_buffer_tmp_0.getval(row_kernel, col_kernel);
						data_t d_1 = window_buffer_tmp_1.getval(row_kernel, col_kernel);
						data_t d_2 = window_buffer_tmp_2.getval(row_kernel, col_kernel);
						data_t d_3 = window_buffer_tmp_3.getval(row_kernel, col_kernel);

						if (no_in_channel_valid == 1)
							d_out += w_0*d_0;
						else if (no_in_channel_valid == 2)
							d_out += w_0*d_0 + w_1*d_1;
						else if (no_in_channel_valid == 3)
							d_out += w_0*d_0 + w_1*d_1 + w_2*d_2;
						else
							d_out += w_0*d_0 + w_1*d_1 + w_2*d_2 + w_3*d_3;
					}
				}
			}

			u32 output_index = n*param_output_h *param_output_w *param_output_c + ch_out*param_output_h*param_output_w + row_out*param_output_w + col_out;
			data_t d_tm_in = data_output_in[output_index];
			data_t d_tmp_out = d_out + d_tm_in;

			if (ch_in == 0)
				d_tm_in = weights_bias_local[ch_out];
			else
				d_tm_in = 0;


			data_output_out_stream << d_tm_in + d_tmp_out;
		}
	}


output_actor_2:
	for (param_t row_out = 0; row_out < param_output_h; row_out++) {
#pragma HLS LOOP_TRIPCOUNT min=_output_h_ max=_output_h_ avg=_output_h_
		for (param_t col_out = 0; col_out < param_output_w; col_out++) {
#pragma HLS LOOP_TRIPCOUNT min=_output_w_ max=_output_w_ avg=_output_w_
#pragma HLS PIPELINE
			u32 output_index = n*param_output_h *param_output_w *param_output_c + ch_out*param_output_h*param_output_w + row_out*param_output_w + col_out;
			data_output_out[output_index] = data_output_out_stream.read();
		}
	}

}



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
) {

	weight_t    weights_local[4][KERNEL_HEIGHT_MAX][KERNEL_WIDTH_MAX];
#pragma HLS ARRAY_PARTITION variable=weights_local dim=1 complete
#pragma HLS ARRAY_PARTITION variable=weights_local dim=2 complete
#pragma HLS ARRAY_PARTITION variable=weights_local dim=3 complete

	weight_t    weights_bias_local[NO_OUT_CHANNEL_MAX];

	for (param_t ch_out = 0; ch_out < param_output_c; ch_out++) {
#pragma HLS LOOP_TRIPCOUNT min=64 max=64 avg=64
#pragma HLS PIPELINE
		weights_bias_local[ch_out] = w_bias[ch_out];
	}
	param_t ch_in = 0;
	for (param_t n_index = 0; n_index < n; n_index++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1 avg=1
		for (param_t ch_out = 0; ch_out < param_output_c; ch_out++) {
#pragma HLS LOOP_TRIPCOUNT min=_output_c_ max=_output_c_ avg=_output_c_
			for (param_t ch_in = 0; ch_in < param_input_c; ch_in += 4) {
#pragma HLS LOOP_TRIPCOUNT min=_input_c_ max=_input_c_ avg=_input_c_

				read_weight(
					w,
					weights_local,

					param_input_h,
					param_input_w,
					param_input_c,
					param_input_s,

					param_output_h,
					param_output_w,
					param_output_c,
					param_output_s,

					param_kernel_h,
					param_kernel_w,
					ch_out,
					ch_in);
				conv_layer_core(
					data_input_0,
					data_input_1,
					data_input_2,
					data_input_3,

					data_output_in,
					data_output_out,

					weights_local,
					weights_bias_local,

					n_index,

					param_input_h,
					param_input_w,
					param_input_c,
					param_input_s,

					param_output_h,
					param_output_w,
					param_output_c,
					param_output_s,

					output_p,

					param_kernel_h,
					param_kernel_w,

					ch_out,
					ch_in

				);
			}
		}
	}
}









/*
#include "caffe/fpga_accel_basic.hpp"
#include "vivado_hls/hls_stream.h"
#include "caffe/layers/fpga_hls_video_mem.hpp"


void read_weight(
	weight_t*    w,
	weight_t     weights_local[NO_OUT_CHANNEL_MAX][KERNEL_HEIGHT_MAX][KERNEL_WIDTH_MAX],
	param_t          param_input_h,
	param_t          param_input_w,
	param_t          param_input_c,
	param_t          param_input_s,

	param_t          param_output_h,
	param_t          param_output_w,
	param_t          param_output_c,
	param_t          param_output_s,

	param_t          param_kernel_h,
	param_t          param_kernel_w,
	param_t          ch_in) {

	for (param_t ch_out = 0; ch_out < param_output_c; ch_out++) {
#pragma HLS LOOP_TRIPCOUNT min=64 max=64 avg=64
		for (param_t row_krn = 0; row_krn < param_kernel_h; row_krn++) {
#pragma HLS LOOP_TRIPCOUNT min=3 max=3 avg=3
			for (param_t col_krn = 0; col_krn < param_kernel_w; col_krn++) {
#pragma HLS LOOP_TRIPCOUNT min=3 max=3 avg=3
#pragma HLS PIPELINE
				int index_tmp = ch_out*(param_kernel_h*param_kernel_w*param_input_c) + ch_in*(param_kernel_h*param_kernel_w) + row_krn*param_kernel_w + col_krn;
				weight_t w_tmp = w[index_tmp];
				weights_local[ch_out][row_krn][col_krn] = w_tmp;
					
			}
		}
	}
}

void conv_layer_core(
	data_t        *data_input,
	data_t        *data_output_in,
	data_t        *data_output_out,
	weight_t       weights_local[NO_OUT_CHANNEL_MAX][KERNEL_HEIGHT_MAX][KERNEL_WIDTH_MAX],
	weight_t       weights_bias_local[NO_OUT_CHANNEL_MAX],

	param_t          n,

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
	param_t          param_kernel_w,
	param_t            ch_in) {

	int tmp_counter_1 = 0;
	int tmp_counter_2 = 0;
	hls::LineBuffer<KERNEL_HEIGHT_MAX - 1, DATA_WIDTH_MAX + (KERNEL_WIDTH_MAX / 2) * 2, data_t> input_buffer;
	hls::Window<KERNEL_HEIGHT_MAX, KERNEL_WIDTH_MAX, data_t>                            window_buffer;

	u32 mask_p = param_kernel_h / 2;

#pragma HLS DATAFLOW
	hls::stream<data_t> input_data_stream;
	hls::stream<data_t> output_data_stream;
	hls::stream<data_t> data_output_out_stream;
	hls::stream<hls::Window<KERNEL_HEIGHT_MAX, KERNEL_WIDTH_MAX, data_t> > window_buffer_stream;
	hls::Window<KERNEL_HEIGHT_MAX, KERNEL_WIDTH_MAX, data_t>  window_buffer_tmp;

read_actor:
	for (u32 row_in = 0; row_in < param_input_h; row_in++) {
#pragma HLS LOOP_TRIPCOUNT min=256 max=256 avg=256
		for (u32 col_in = 0; col_in < param_input_w; col_in++) {
#pragma HLS LOOP_TRIPCOUNT min=256 max=256 avg=256

#pragma HLS PIPELINE

			input_data_stream << data_input[n*param_input_h *param_input_w *param_input_c + row_in*param_input_w*param_input_c + col_in*param_input_c + ch_in];

		}
	}


	//filter actor
	//===========================================================
	// input zero padding, here we assume kernel is of size 3x3
	//===========================================================
window_slide_actor:
	for (param_t row_in = 0; row_in < param_input_h + (output_p) * 2; row_in++) {
#pragma HLS LOOP_TRIPCOUNT min=258 max=258 avg=258
		for (param_t col_in = 0; col_in < param_input_w + (output_p) * 2; col_in++) {
#pragma HLS LOOP_TRIPCOUNT min=258 max=258 avg=258

#pragma HLS PIPELINE
			data_t  d_in;
			if ((row_in >= (output_p) && row_in < param_input_h + (output_p)) && (col_in >= (output_p) && col_in < param_input_w + (output_p))) {
				d_in = input_data_stream.read();
				//				d_in = data_input[(row_in-param_kernel_h/2)*param_input_w*param_input_c+(col_in-param_kernel_w/2)*param_input_c + ch_in];
			}
			else {
				d_in = 0;  //zero padding value
			}


			data_t  d_tmp;

			if (row_in == 0)
				d_tmp = 0;
			else
				d_tmp = input_buffer.getval(0, col_in);

			//=====================Update Line buffer=
			input_buffer.shift_pixels_up(col_in);
			input_buffer(param_kernel_h - 2, col_in) = d_in;
			//========================================


			//=====================Update window data=
			window_buffer.shift_left();

			window_buffer.insert(d_tmp, 0, param_kernel_w - 1);
			for (param_t row_wndw = 1; row_wndw < KERNEL_HEIGHT_MAX - 1; row_wndw++) {
				if (row_wndw < param_kernel_w - 1)
					window_buffer.insert(input_buffer.getval(row_wndw - 1, col_in), row_wndw, param_kernel_w - 1);
			}

			window_buffer.insert(d_in, param_kernel_w - 1, param_kernel_w - 1);
			//========================================

			int row_in_update = row_in - (mask_p*2);
			int col_in_update = col_in - (mask_p*2);
			if ((row_in_update >= 0 )&& row_in_update%param_output_s == 0 &&
				col_in_update >= 0 && col_in_update%param_output_s == 0
				) {
//				tmp_counter_1++;
				window_buffer_stream << window_buffer;
			}

		}
	}

//	std::cout << "tmp_counter_1 = " << tmp_counter_1 << std::endl;

filter_actor:
	for (param_t row_out = 0; row_out < param_output_h; row_out++) {
#pragma HLS LOOP_TRIPCOUNT min=128 max=128 avg=128
		for (param_t col_out = 0; col_out < param_output_w; col_out++) {
#pragma HLS LOOP_TRIPCOUNT min=128 max=128 avg=128
			for (param_t ch_out = 0; ch_out < param_output_c; ch_out++) {
#pragma HLS LOOP_TRIPCOUNT min=64 max=64 avg=64

#pragma HLS PIPELINE
				if (ch_out == 0) {
//					tmp_counter_2++;
					window_buffer_tmp = window_buffer_stream.read();
				}
				data_t d_out = 0;
				for (param_t row_kernel = 0; row_kernel < KERNEL_HEIGHT_MAX; row_kernel++) {
					for (param_t col_kernel = 0; col_kernel < KERNEL_WIDTH_MAX; col_kernel++) {
						if (row_kernel < param_kernel_h  && col_kernel < param_kernel_w) {
							data_t w = weights_local[ch_out][row_kernel][col_kernel];
							data_t d = window_buffer_tmp.getval(row_kernel, col_kernel);
							d_out += w*d;
						}
					}
				}
				//				output_data_stream << d_out;

				u32 output_index = n*param_output_h *param_output_w *param_output_c + row_out*param_output_w*param_output_c + col_out*param_output_c + ch_out;
				data_t d_tm_in = data_output_in[output_index];
				data_t d_tmp_out = d_out;

				if (ch_in == 0) {
					d_tm_in = weights_bias_local[ch_out];
				}

				//data_output_out[output_index] = d_tm_in + d_tmp_out;
				data_output_out_stream << d_tm_in + d_tmp_out;

			}
		}
	}
//	std::cout << "tmp_counter_2 = " << tmp_counter_2 << std::endl;
output_actor_2:
	for (param_t row_out = 0; row_out < param_output_h; row_out++) {
#pragma HLS LOOP_TRIPCOUNT min=128 max=128 avg=128
		for (param_t col_out = 0; col_out < param_output_w; col_out++) {
#pragma HLS LOOP_TRIPCOUNT min=128 max=128 avg=128
			for (param_t ch_out = 0; ch_out < param_output_c; ch_out++) {
#pragma HLS LOOP_TRIPCOUNT min=64 max=64 avg=64

#pragma HLS PIPELINE
				u32 output_index = n*param_output_h *param_output_w *param_output_c + row_out*param_output_w*param_output_c + col_out*param_output_c + ch_out;
				data_output_out[output_index] = data_output_out_stream.read();
			}
		}
	}

}



void fpga_accel_cnn(
	data_t        data_input[DATA_HEIGHT_MAX*DATA_WIDTH_MAX],
	data_t        data_output[DATA_HEIGHT_MAX*DATA_WIDTH_MAX],
	weight_t      w[NO_OUT_CHANNEL_MAX*NO_IN_CHANNEL_MAX*3 * 3],
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

	param_t       output_p,

	param_t       param_kernel_h,
	param_t       param_kernel_w
) {

#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE m_axi depth=1024 port=data_input   offset=slave          bundle=gmem_0
#pragma HLS INTERFACE m_axi depth=1024 port=data_output  offset=slave          bundle=gmem_1
#pragma HLS INTERFACE m_axi depth=1024 port=w      offset=slave          bundle=gmem_2


#pragma HLS INTERFACE s_axilite port=data_input  bundle=control
#pragma HLS INTERFACE s_axilite port=data_output bundle=control
#pragma HLS INTERFACE s_axilite port=weights     bundle=control

#pragma HLS INTERFACE s_axilite port=param_input  bundle=control
#pragma HLS INTERFACE s_axilite port=param_output bundle=control


	weight_t    weights_local[NO_OUT_CHANNEL_MAX][KERNEL_HEIGHT_MAX][KERNEL_WIDTH_MAX];
#pragma HLS ARRAY_PARTITION variable=weights_local dim=2 complete
#pragma HLS ARRAY_PARTITION variable=weights_local dim=3 complete

	weight_t    weights_bias_local[NO_OUT_CHANNEL_MAX];

	for (param_t ch_out = 0; ch_out < param_output_c; ch_out++) {
#pragma HLS LOOP_TRIPCOUNT min=64 max=64 avg=64
#pragma HLS PIPELINE
		weights_bias_local[ch_out] =  w_bias[ch_out];
	}
	param_t ch_in = 0;
	for (param_t n_index = 0; n_index < n; n_index++) {
		for (param_t ch_in = 0; ch_in < param_input_c; ch_in++) {
#pragma HLS LOOP_TRIPCOUNT min=3 max=3 avg=3
			read_weight(
				w,
				weights_local,

				param_input_h,
				param_input_w,
				param_input_c,
				param_input_s,

				param_output_h,
				param_output_w,
				param_output_c,
				param_output_s,

				param_kernel_h,
				param_kernel_w,
				ch_in);
			conv_layer_core(
				
				data_input,
				data_output,
				data_output,
				weights_local,
				weights_bias_local,

				n_index, 

				param_input_h,
				param_input_w,
				param_input_c,
				param_input_s,

				param_output_h,
				param_output_w,
				param_output_c,
				param_output_s,

				output_p,

				param_kernel_h,
				param_kernel_w,

				ch_in);
		}
	}
}*/
#else
#include "caffe/fpga_accel_driver.hpp"
#include <time.h>
#include <sys/time.h>
#include <iostream>




void fpga_conv_layer(
	data_t        data_input_0[DATA_HEIGHT_MAX*DATA_WIDTH_MAX],
	data_t        data_input_1[DATA_HEIGHT_MAX*DATA_WIDTH_MAX],
	data_t        data_input_2[DATA_HEIGHT_MAX*DATA_WIDTH_MAX],
	data_t        data_input_3[DATA_HEIGHT_MAX*DATA_WIDTH_MAX],

	data_t        data_output_in[DATA_HEIGHT_MAX*DATA_WIDTH_MAX],
	data_t        data_output_out[DATA_HEIGHT_MAX*DATA_WIDTH_MAX],

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

	param_t       output_p,

	param_t       param_kernel_h,
	param_t       param_kernel_w
) {
	double hardware_start;
	double hardware_end;
	double hardware_execution_time;

	accel_write(
		data_input_0,
		data_input_1,
		data_input_2,
		data_input_3,

		data_output_in,
		data_output_out,
		w,
		w_bias,

		n,

		param_input_h,
		param_input_w,
		param_input_c,
		param_input_s,

		param_output_h,
		param_output_w,
		param_output_c,
		param_output_s,

		output_p,

		param_kernel_h,
		param_kernel_w
	);

	hardware_start = getTimestamp();

	accel_start();
	sleep(10);  //interrupt

	hardware_end = getTimestamp();
	hardware_execution_time = (hardware_end - hardware_start) / (1000);
	std::cout << "conv only without memory cache overhead on FPGA layer " <<  " execution time  " << hardware_execution_time << "  ms elapsed\n";

}

#endif // WIN32
