#ifndef CAFFE_BASE_CONVOLUTION_LAYER_HPP_
#define CAFFE_BASE_CONVOLUTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/im2col.hpp"

namespace caffe {

/**
 * @brief Abstract base class that factors out the BLAS code common to
 *        ConvolutionLayer and DeconvolutionLayer.
 */
template <typename Dtype>
class BaseConvolutionLayer : public Layer<Dtype> {
 public:
  explicit BaseConvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }

 protected:

 void forward_cpu_gemm_ccnmm_merge(const Dtype* input,const Dtype* weights,
     Dtype* output, int batch_idx, const vector<Blob<Dtype>*>& bottom,
       bool skip_im2col = false);
 void forward_cpu_gemm_xu_xuv(const Dtype* input,Dtype* input2, const Dtype* weights,const Dtype* weights2,
     Dtype* output, Dtype* output2, int batch_idx,const vector<Blob<Dtype>*>& bottom, bool skip_im2col = false);

  // Helper functions that abstract away the column buffer and gemm arguments.
  // The last argument in forward_cpu_gemm is so that we can skip the im2col if
  // we just called weight_cpu_gemm with the same input.
  void forward_cpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* output, bool skip_im2col = false);
  void forward_cpu_bias(Dtype* output, const Dtype* bias);
  void backward_cpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* output);
  void weight_cpu_gemm(const Dtype* input, const Dtype* output, Dtype*
      weights);
  void backward_cpu_bias(Dtype* bias, const Dtype* input);

#ifndef CPU_ONLY
  void forward_gpu_gemm(const Dtype* col_input, const Dtype* weights,
      Dtype* output, bool skip_im2col = false);
  void forward_gpu_bias(Dtype* output, const Dtype* bias);
  void backward_gpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* col_output);
  void weight_gpu_gemm(const Dtype* col_input, const Dtype* output, Dtype*
      weights);
  void backward_gpu_bias(Dtype* bias, const Dtype* input);
#endif

  /// @brief The spatial dimensions of the input.
  inline int input_shape(int i) {
    return (*bottom_shape_)[channel_axis_ + i];
  }
  // reverse_dimensions should return true iff we are implementing deconv, so
  // that conv helpers know which dimensions are which.
  virtual bool reverse_dimensions() = 0;
  // Compute height_out_ and width_out_ from other parameters.
  virtual void compute_output_shape() = 0;

  /// @brief The spatial dimensions of a filter kernel.
  Blob<int> kernel_shape_;
  /// @brief The spatial dimensions of the stride.
  Blob<int> stride_;
  /// @brief The spatial dimensions of the padding.
  Blob<int> pad_;
  /// @brief The spatial dimensions of the dilation.
  Blob<int> dilation_;
  /// @brief The spatial dimensions of the convolution input.
  Blob<int> conv_input_shape_;
  /// @brief The spatial dimensions of the col_buffer.
  vector<int> col_buffer_shape_;
  vector<int> col_buffer_shape_xu;
  vector<int> col_buffer_shape_xuv;
  vector<int> col_buffer_shape_csrmm;
  /// @brief The spatial dimensions of the output.
  vector<int>col_buffer_shape_ccnmm;
  /// @brief The spatial dimensions of the output.
  vector<int> output_shape_;
  vector<int> xu_shape_;
  vector<int> xuv_shape_;
  //buffer for step = 3
  Blob<Dtype> xu_buffer_;
  Blob<Dtype> xuv_buffer_;
  Blob<Dtype> xs_buffer_;
  const vector<int>* bottom_shape_;

  int num_spatial_axes_;
  int bottom_dim_;
  int top_dim_;

  int channel_axis_;
  int num_;
  int channels_;
  int group_;
  int out_spatial_dim_;
  int weight_offset_;
  int num_output_;
  bool bias_term_;
  bool is_1x1_;
  bool force_nd_im2col_;

  int initial_check;
  Blob<Dtype> transposed_output_buffer_;

  Dtype step;
  Blob<int> col_buf_mask_;
  Blob<int> col_buf_mask_ccnmm;
  Blob<int> row_buf_mask_ccnmm;
  vector<int> left_columns_;//the number of left columns of weight matrix for each group
  vector<int> left_rows_; //the number of left rows of weight matrix for each group
  Blob<Dtype> squeezed_weight_buffer_;
  Blob<Dtype> squeezed_weight_buffer_ccnmm;


  Blob<Dtype> col_buffer_;
  Blob<Dtype> bias_multiplier_;
  Blob<Dtype> col_buffer_csrmm;
  Blob<Dtype> col_buffer_ccnmm;
  //for step = 3
  Blob<Dtype> col_buffer_xu;
  Blob<Dtype> col_buffer_xuv;

 //private:
protected:
   //TODO: the parameter of kernel v should be read from prototxt, should be fixed later
   inline void conv_im2col_cpu_xuv(const Dtype* data, Dtype* col_buff,int* all_zero_mask = NULL) {

     if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
       im2col_cpu(data, conv_in_channels_,
           conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
           1, 1,
           0, 0,
           stride_.cpu_data()[0], stride_.cpu_data()[1],
           dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff,
           all_zero_mask);
     } else {
       // JSP: FIXME - parallelization over multiple inputs in a batch probably
       // won't work for this code path
       im2col_nd_cpu(data, num_spatial_axes_, conv_input_shape_.cpu_data(),
           col_buffer_shape_.data(), kernel_shape_.cpu_data(),
           pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), col_buff);
     }
   }

   inline void conv_im2col_cpu_xuv_nc(Dtype* data, Dtype* col_buff,int* all_zero_mask = NULL) {

     if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
       im2col_cpu(data, conv_in_channels_,
           conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
           1, 1,
           0, 0,
           stride_.cpu_data()[0], stride_.cpu_data()[1],
           dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff,
           all_zero_mask);
     } else {
       // JSP: FIXME - parallelization over multiple inputs in a batch probably
       // won't work for this code path
       im2col_nd_cpu(data, num_spatial_axes_, conv_input_shape_.cpu_data(),
           col_buffer_shape_.data(), kernel_shape_.cpu_data(),
           pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), col_buff);
     }
   }


   inline void conv_im2col_cpu_xu(const Dtype* data, Dtype* col_buff,int* all_zero_mask = NULL) {

     if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
    //   LOG(INFO)<<"kernel parameter: "<< stride_.cpu_data()[0]<<","<< stride_.cpu_data()[1]<<","<<
        //dilation_.cpu_data()[0]<<","<<dilation_.cpu_data()[1];
       im2col_cpu(data, conv_in_channels_,
           conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
           kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
           pad_.cpu_data()[0], pad_.cpu_data()[1],
           stride_.cpu_data()[0], stride_.cpu_data()[1],
           dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff,
           all_zero_mask);
     } else {
       // JSP: FIXME - parallelization over multiple inputs in a batch probably
       // won't work for this code path
       im2col_nd_cpu(data, num_spatial_axes_, conv_input_shape_.cpu_data(),
           col_buffer_shape_.data(), kernel_shape_.cpu_data(),
           pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), col_buff);
     }
   }
  // wrap im2col/col2im so we don't have to remember the (long) argument lists
  inline void conv_im2col_cpu(const Dtype* data, Dtype* col_buff,int* all_zero_mask = NULL) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      im2col_cpu(data, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff,all_zero_mask);
    } else {
      im2col_nd_cpu(data, num_spatial_axes_, conv_input_shape_.cpu_data(),
          col_buffer_shape_.data(), kernel_shape_.cpu_data(),
          pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), col_buff);
    }
  }
  inline void conv_col2im_cpu(const Dtype* col_buff, Dtype* data) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      col2im_cpu(col_buff, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
    } else {
      col2im_nd_cpu(col_buff, num_spatial_axes_, conv_input_shape_.cpu_data(),
          col_buffer_shape_.data(), kernel_shape_.cpu_data(),
          pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), data);
    }
  }
#ifndef CPU_ONLY
  inline void conv_im2col_gpu(const Dtype* data, Dtype* col_buff) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      im2col_gpu(data, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
    } else {
      im2col_nd_gpu(data, num_spatial_axes_, num_kernels_im2col_,
          conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
          kernel_shape_.gpu_data(), pad_.gpu_data(),
          stride_.gpu_data(), dilation_.gpu_data(), col_buff);
    }
  }
  inline void conv_col2im_gpu(const Dtype* col_buff, Dtype* data) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      col2im_gpu(col_buff, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
    } else {
      col2im_nd_gpu(col_buff, num_spatial_axes_, num_kernels_col2im_,
          conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
          kernel_shape_.gpu_data(), pad_.gpu_data(), stride_.gpu_data(),
          dilation_.gpu_data(), data);
    }
  }
#endif

Blob<Dtype> nz_weight_values_;//nonzero elements
Blob<int> nz_weight_indices_;//index of nonzero
Blob<int> nz_weight_index_pointers_;//pointer(index) of indices
bool is_concatenating_weights_features_; //if use concatenation scheme to compress dense weights and features together
Blob<int> dense_feature_map_mask_;//to skip all zero rows in col_buffer_
Blob<int> nz_per_row_;//nonzero per row for cusparse
vector<int> nz_num_;//the number of nonzero for cusparse


  int num_kernels_im2col_;
  int num_kernels_col2im_;
  int conv_out_channels_;
  int conv_in_channels_;
  int conv_out_spatial_dim_;
  int kernel_dim_;
  int col_offset_;
  int output_offset_;

};

}  // namespace caffe

#endif  // CAFFE_BASE_CONVOLUTION_LAYER_HPP_
