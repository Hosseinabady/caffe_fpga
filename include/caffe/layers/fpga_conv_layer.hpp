#ifndef CAFFE_FPGA_CONV_LAYER_HPP_
#define CAFFE_FPGA_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

#ifdef USE_FPGA
	
	template <typename Dtype>
	class FPGAConvolutionLayer : public ConvolutionLayer<Dtype> {
	public:
//		explicit FPGAConvolutionLayer(const LayerParameter& param)
//			: ConvolutionLayer<Dtype>(param), handles_setup_(false) {}
		virtual ~FPGAConvolutionLayer() {};

	protected:
		virtual void Forward_fpga(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
//		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
//			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

//		bool handles_setup_;
//		cudnnHandle_t* handle_;
//		cudaStream_t*  stream_;

		// algorithms for forward and backwards convolutions
//		cudnnConvolutionFwdAlgo_t *fwd_algo_;
//		cudnnConvolutionBwdFilterAlgo_t *bwd_filter_algo_;
//		cudnnConvolutionBwdDataAlgo_t *bwd_data_algo_;

//		vector<cudnnTensorDescriptor_t> bottom_descs_, top_descs_;
//		cudnnTensorDescriptor_t    bias_desc_;
//		cudnnFilterDescriptor_t      filter_desc_;
//		vector<cudnnConvolutionDescriptor_t> conv_descs_;
//		int bottom_offset_, top_offset_, bias_offset_;

//		size_t *workspace_fwd_sizes_;
//		size_t *workspace_bwd_data_sizes_;
//		size_t *workspace_bwd_filter_sizes_;
//		size_t workspaceSizeInBytes;  // size of underlying storage
//		void *workspaceData;  // underlying storage
//		void **workspace;  // aliases into workspaceData
	};
#endif

}  // namespace caffe

#endif  // CAFFE_FPGA_CONV_LAYER_HPP_