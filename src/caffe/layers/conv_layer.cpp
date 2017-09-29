#include <vector>

#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/fpga_conv_layer.hpp"
#include "caffe/layers/fpga_accel_cnn_layers.hpp"

#ifndef WIN32
#include <sys/time.h>
#include <time.h>

#include "caffe/fpga_accel_driver.hpp"
#endif// WIN32


int approximate_weight(float *a, int n);
namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
//	std::cout << "void ConvolutionLayer<Dtype>::Forward_cpu" << std::endl;

  Dtype* weight = this->blobs_[0]->mutable_cpu_data();

  
  if (this->blobs_[0]->height() > 1) {
//	  approximate_weight((float*)weight, this->blobs_[0]->height());
  }

  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
//	int  n = 0;
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}
#ifdef USE_FPGA
template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_fpga(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	std::cout << "void ConvolutionLayer<Dtype>::Forward_fpga" << std::endl;
//	Forward_cpu(bottom, top);
	
//	int bottom_size = bottom.size();
	for (int i = 0; i < bottom.size(); ++i) {




		const Dtype* weight = this->blobs_[0]->fpga_data();
		const Dtype* bias = this->blobs_[1]->fpga_data();


		std::cout << "************** address bias = " << bias << std::endl;
		std::cout << "************** address weight = " << weight << std::endl;


		data_t* input_data = (data_t*)bottom[i]->fpga_data();
		std::cout << "************** address input_data = " << input_data << std::endl;
		data_t* output_data = (data_t*)top[i]->fpga_data();
		std::cout << "************** address output_data = " << output_data << std::endl;



		int input_h = bottom[i]->height();
		int input_w = bottom[i]->width();
		int input_c = bottom[i]->channels();
		int input_s = 1;


		const int* stride_data = this->stride_.cpu_data();
		const int* pad_data = this->pad_.cpu_data();

		int output_h = top[i]->height();
		int output_w = top[i]->width();
		int output_c = top[i]->channels();
		int output_s = stride_data[0];

		int output_p = pad_data[0];



		int kernel_h = this->blobs_[0]->height();
		int kernel_w = this->blobs_[0]->width();

#ifndef WIN32
		double hardware_start;
		double hardware_end;
		double hardware_execution_time;


		hardware_start = getTimestamp();
#endif// WIN32	
		fpga_conv_layer(
			input_data,
			input_data,
			input_data,
			input_data,


			output_data,
			output_data,
			(weight_t*)weight,
			(weight_t*)bias,

			this->num_,

			input_h,
			input_w,
			input_c,
			input_s,



			output_h,
			output_w,
			output_c,
			output_s,

			output_p,

			kernel_h,
			kernel_w);
#ifndef WIN32
		hardware_end = getTimestamp();
		hardware_execution_time = (hardware_end - hardware_start) / (1000);
		if (Caffe::mode() == Caffe::FPGA)
			std::cout << "Only Conv layer on FPGA layer " << " execution time  " << hardware_execution_time << "  ms elapsed\n";
		else
			std::cout << "on ARM layer " << " execution time  " << hardware_execution_time << "  ms elapsed\n";

#endif// WIN32	

	}
	
}
#endif// USE_FPGA
template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
