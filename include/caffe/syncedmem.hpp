#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>

#ifdef USE_MKL
  #include "mkl.h"
#endif

#include "caffe/common.hpp" 


#ifdef USE_FPGA //Added by Mohammad
#include "caffe/fpga.hpp" //Added by Mohammad
#endif //USE_FPGA //Added by Mohammad

namespace caffe {

// If CUDA is available and in GPU mode, host memory will be allocated pinned,
// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
// The improvement in performance seems negligible in the single GPU case,
// but might be more significant for parallel training. Most importantly,
// it improved stability for large models on many GPUs.
inline void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda, bool* use_fpga) {
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaMallocHost(ptr, size));
    *use_cuda = true;
	*use_fpga = false;                                   //Added by Mohammad
    return;
  }
#endif

#ifdef USE_FPGA                                        //Added by Mohammad
  if (Caffe::mode() == Caffe::FPGA && *use_fpga == true) {                    //Added by Mohammad
	  *ptr = fpga_malloc(size);                          //Added by Mohammad
//	  std::cout << "==================fpga malloc  addrss  " << ptr << " size = " << size << std::endl;
	  *use_cuda = false;                                 //Added by Mohammad
	  *use_fpga = true;                                  //Added by Mohammad
	  return;                                            //Added by Mohammad
  }                                                      //Added by Mohammad
#endif    //USE_FPGA                                                //Added by Mohammad
#ifdef USE_MKL
  *ptr = mkl_malloc(size ? size:1, 64);
#else
  *ptr = malloc(size);
//  std::cout << "==================cpu malloc  addrss  " << ptr << " size = " << size << std::endl;
#endif
  *use_cuda = false;
  *use_fpga = false;                                     //Added by Mohammad
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}

inline void CaffeFreeHost(void* ptr, bool use_cuda, bool* use_fpga) {
#ifndef CPU_ONLY
  if (use_cuda) {
    CUDA_CHECK(cudaFreeHost(ptr));
    return;
  }
#endif

#ifdef USE_FPGA                                        //Added by Mohammad
  if (Caffe::mode() == Caffe::FPGA && *use_fpga == true) {                    //Added by Mohammad
	  fpga_free(ptr);                                    //Added by Mohammad
	  return;                                            //Added by Mohammad
  }                                                      //Added by Mohammad
#endif //USE_FPGA
#ifdef USE_MKL
  mkl_free(ptr);
#else
  free(ptr);
#endif
}


/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 *
 * TODO(dox): more thorough description.
 */
class SyncedMemory {
 public:
  SyncedMemory();
  explicit SyncedMemory(size_t size);
  ~SyncedMemory();
  const void* cpu_data();
  void set_cpu_data(void* data);
  const void* gpu_data();
  void set_gpu_data(void* data);
  void* mutable_cpu_data();
  void* mutable_gpu_data();

  const void* fpga_data();
  void set_fpga_data(void* data);
  void* mutable_fpga_data();

  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, HEAD_AT_FPGA, SYNCED };
  SyncedHead head() { return head_; }
  size_t size() { return size_; }

#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif

 private:
  void check_device();

  void to_cpu();
  void to_fpga();
  void to_gpu();
  void* cpu_ptr_;
  void* gpu_ptr_;
  size_t size_;
  SyncedHead head_;
  bool own_cpu_data_;
  bool cpu_malloc_use_cuda_;
  bool own_gpu_data_;
  bool cpu_malloc_use_fpga_;
  int device_;

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
