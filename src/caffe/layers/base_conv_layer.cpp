#include <algorithm>
#include <vector>
#include <omp.h>

#include "caffe/filler.hpp"
#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  initial_check = 1;
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  force_nd_im2col_ = conv_param.force_nd_im2col();
  channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
  const int first_spatial_axis = channel_axis_ + 1;
  const int num_axes = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(num_spatial_axes_, 0);
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));
  // Setup filter kernel dimensions (kernel_shape_).
  kernel_shape_.Reshape(spatial_dim_blob_shape);
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "kernel_h & kernel_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.kernel_size_size())
        << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = conv_param.kernel_h();
    kernel_shape_data[1] = conv_param.kernel_w();
  } else {
    const int num_kernel_dims = conv_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
        << "kernel_size must be specified once, or once per spatial dimension "
        << "(kernel_size specified " << num_kernel_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
      for (int i = 0; i < num_spatial_axes_; ++i) {
        kernel_shape_data[i] =
            conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
      }
  }
  for (int i = 0; i < num_spatial_axes_; ++i) {
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }
  // Setup stride dimensions (stride_).
  stride_.Reshape(spatial_dim_blob_shape);
  int* stride_data = stride_.mutable_cpu_data();
  if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = conv_param.stride_h();
    stride_data[1] = conv_param.stride_w();
  } else {
    const int num_stride_dims = conv_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultStride = 1;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
          conv_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }
  // Setup pad dimensions (pad_).
  pad_.Reshape(spatial_dim_blob_shape);
  int* pad_data = pad_.mutable_cpu_data();
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = conv_param.pad_h();
    pad_data[1] = conv_param.pad_w();
  } else {
    const int num_pad_dims = conv_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultPad = 0;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          conv_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // Setup dilation dimensions (dilation_).
  dilation_.Reshape(spatial_dim_blob_shape);
  int* dilation_data = dilation_.mutable_cpu_data();
  const int num_dilation_dims = conv_param.dilation_size();
  CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
        num_dilation_dims == num_spatial_axes_)
      << "dilation must be specified once, or once per spatial dimension "
      << "(dilation specified " << num_dilation_dims << " times; "
      << num_spatial_axes_ << " spatial dims).";
  const int kDefaultDilation = 1;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                       conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = true;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    is_1x1_ &=
        kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
    if (!is_1x1_) { break; }
  }
  // Configure output channels and groups.
  channels_ = bottom[0]->shape(channel_axis_);
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.convolution_param().group();
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  if (reverse_dimensions()) {
    conv_out_channels_ = channels_;
    conv_in_channels_ = num_output_;
  } else {
    conv_out_channels_ = num_output_;
    conv_in_channels_ = channels_;
  }
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  vector<int> weight_shape(2);
  weight_shape[0] = conv_out_channels_;
  weight_shape[1] = conv_in_channels_ / group_;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    weight_shape.push_back(kernel_shape_data[i]);
  }
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  vector<int> bias_shape(bias_term_, num_output_);
  if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + bias_term_, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
          << weight_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[0]->shape_string();
    }
    if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
          << bias_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  kernel_dim_ = this->blobs_[0]->count(1);
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
step = this->layer_param_.sparse_param().step();

  const int first_spatial_axis = channel_axis_ + 1;
  CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
      << "bottom num_axes may not change.";
  num_ = bottom[0]->count(0, channel_axis_);
  CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)
      << "Input size incompatible with convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    if(step != 3){
    //  LOG(INFO)<<"step"<<step;
    CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
        << "All inputs must have the same shape.";
    }
  }
  // Shape the tops.

  if(step == 0){
      bottom_shape_ = &bottom[0]->shape();
      compute_output_shape();
      vector<int> top_shape(bottom[0]->shape().begin(),
          bottom[0]->shape().begin() + channel_axis_);
      top_shape.push_back(num_output_);
      for (int i = 0; i < num_spatial_axes_; ++i) {
        top_shape.push_back(output_shape_[i]);
      }
      for (int top_id = 0; top_id < top.size(); ++top_id) {
        top[top_id]->Reshape(top_shape);
      }
  }
  else if(step == 3 ) {
    if(bottom.size() != 3) LOG(FATAL)<<"Sparse layer step 3 should be have x,s,v three bottoms";
  //same as step = 1
  //NOTICE: the top[0] size is decided by bottom[1] sparse size
  //as the kernel u does not decide output size
  bottom_shape_ = &bottom[0]->shape();
  compute_output_shape();
  vector<int> top_shape(bottom[0]->shape().begin(),
      bottom[0]->shape().begin() + channel_axis_);
  vector<int> xu_shape(bottom[0]->shape().begin(),
      bottom[0]->shape().begin() + channel_axis_);
  //vector<int> xuv_shape(bottom[0]->shape().begin(),
    //  bottom[0]->shape().begin() + channel_axis_);
  //vector<int> xs_shape(bottom[0]->shape().begin(),
    //  bottom[0]->shape().begin() + channel_axis_);
    xu_shape.push_back(bottom[1]->shape(0));
    //LOG(INFO)<<"xu_shape: "<< bottom[1]->shape(0);
    //LOG(INFO)<<"num_output_"<<num_output_;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      xu_shape.push_back(output_shape_[i]);
      //LOG(INFO)<<"xu_shape_: "<< output_shape_[i];
    }
    //reshape xu size
    xu_buffer_.Reshape(xu_shape);

  //====================================================//

    //  top_sparse_shape.push_back(bottom[1]->shape(0));
/*
    const int* stride_data = this->stride_.cpu_data();
    const int* pad_data = this->pad_.cpu_data();
    const int* dilation_data = this->dilation_.cpu_data();
*/
    top_shape.push_back(num_output_);
    //LOG(INFO)<<"top, xuv,xs shape: "<<num_output_;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      top_shape.push_back(output_shape_[i]);
    //  LOG(INFO)<<"top, xuv,xs shape: "<<output_shape_[i];
    }

    //top, xuv, xs size are same
    top[0]->Reshape(top_shape);
    xuv_buffer_.Reshape(top_shape);
    xs_buffer_.Reshape(top_shape);

  }

  if (reverse_dimensions()) {
    conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
  } else {
    conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
  }
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
  // Setup input dimensions (conv_input_shape_).
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  conv_input_shape_.Reshape(bottom_dim_blob_shape);
  int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
  for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
    if (reverse_dimensions()) {
      conv_input_shape_data[i] = top[0]->shape(channel_axis_ + i);
    } else {
      conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
    }
  }
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  col_buffer_shape_.clear();
  col_buffer_shape_.push_back(kernel_dim_ * group_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    if (reverse_dimensions()) {
      col_buffer_shape_.push_back(input_shape(i + 1));
    } else {
      col_buffer_shape_.push_back(output_shape_[i]);
    }
  }

  if(step == 3 && Caffe::mode() == Caffe::CPU)
  {
    col_buffer_shape_xu.clear();
    col_buffer_shape_xu.push_back(bottom[1]->count(1) * group_ * num_);
    col_buffer_shape_xu.push_back(xu_buffer_.shape(2));
    col_buffer_shape_xu.push_back(xu_buffer_.shape(3));
    col_buffer_shape_xuv.clear();
    col_buffer_shape_xuv.push_back(bottom[2]->count(1) * group_ * num_);
    col_buffer_shape_xuv.push_back(xuv_buffer_.shape(2));
    col_buffer_shape_xuv.push_back(xuv_buffer_.shape(3));

  //LOG(INFO)<<"col_buffer_shape_xu: "<<bottom[1]->count(1) * group_ * num_<<", "<<xu_buffer_.shape(2)<<", "<<xu_buffer_.shape(3);
  //LOG(INFO)<<"col_buffer_shape_xuv: "<<bottom[2]->count(1) * group_ * num_<<", "<<xuv_buffer_.shape(2)<<", "<<xuv_buffer_.shape(3);

    col_buffer_xu.Reshape(col_buffer_shape_xu);
    col_buffer_xuv.Reshape(col_buffer_shape_xuv);
  }

  col_buffer_.Reshape(col_buffer_shape_);

  if(step == 3 && Caffe::mode() == Caffe::CPU){
   if(step == 3)
    {
      //initial the size of col_buffer_shape_ccnmm for CCNMM operator
        col_buffer_shape_ccnmm.clear();
      //  LOG(INFO)<<"check point 1, kernel_dim_: "<<bottom[1]->count(1)<<"num_"<<bottom[0]->count(1);
        col_buffer_shape_ccnmm.push_back(this->blobs_[0]->count(1) * group_ * bottom[0]->count(0,1)); //col_buffer_shape_.push_back(kernel_dim_ * group_ * num_);
      //  LOG(INFO)<<"check point 2";
        col_buffer_shape_ccnmm.push_back(xs_buffer_.shape(2));//output_h
      //  LOG(INFO)<<"check point 3";
        col_buffer_shape_ccnmm.push_back(xs_buffer_.shape(3));//output_w
      //  LOG(INFO)<<"check point 4";
        col_buffer_ccnmm.Reshape(col_buffer_shape_ccnmm);
    }
  }

if(step == 3)    {
  col_buf_mask_ccnmm.Reshape(1,1,1,this->blobs_[0]->count(1)*group_);
  row_buf_mask_ccnmm.Reshape(1,1,1,bottom[1]->shape(0));
}

  if(!reverse_dimensions()){
    col_buf_mask_.Reshape(1,1,1,kernel_dim_*group_);

    if(step == 3)
    {
        nz_weight_values_.Reshape(1, 1, 1, this->blobs_[0]->count());//nonzero elements
        nz_weight_indices_.Reshape(1,1,1,nz_weight_values_.count());//index of nonzero
        nz_weight_index_pointers_.Reshape(1,1,1,this->blobs_[0]->shape(0)+group_);//pointer(index) of indices
        nz_per_row_.Reshape(1,1,1,this->blobs_[0]->shape(0));
        nz_num_.resize(group_);
        transposed_output_buffer_.Reshape(1,1,xs_buffer_.count(2),this->blobs_[0]->shape(0)/group_);
    }
  }

  bottom_dim_ = bottom[0]->count(channel_axis_);
  top_dim_ = top[0]->count(channel_axis_);
  num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
  num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  out_spatial_dim_ = top[0]->count(first_spatial_axis);
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, out_spatial_dim_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }
}

template <>
void BaseConvolutionLayer<double>::forward_cpu_gemm_xu_xuv(const double* input,double* input_xu,
    const double* weights,const double* weights2, double* output, double* output2,int batch_idx, const vector<Blob<double>*>& bottom,
    bool skip_im2col) {
  NOT_IMPLEMENTED;
}

extern double padding_time, im2col_time;

template<>
void BaseConvolutionLayer<float>::forward_cpu_gemm_xu_xuv(const float* input,float* input_xu,
    const float* weights,const float* weights2, float* output, float* output2, int batch_idx, const vector<Blob<float>*>& bottom,
    bool skip_im2col) {
  const float* col_buff = input;

  float *input_padded;
  int input_padded_len;
//xu setting
  if (!is_1x1_ ||  is_concatenating_weights_features_) {
   //if (tid == 0) im2col_time -= omp_get_wtime();
   int offset = 0;
   if (!skip_im2col || is_concatenating_weights_features_) {
     offset = bottom[1]->count(1)*xu_buffer_.count(2)*group_*batch_idx;
     float *col_buff_mutable = col_buffer_xu.mutable_cpu_data() + offset;
       conv_im2col_cpu_xu(input, col_buff_mutable);
   }
   col_buff = col_buffer_xu.cpu_data() + offset;
   //LOG(INFO)<<"xu :col_buffer_ size: "<<col_buffer_xu.shape(0)<<","<<col_buffer_xu.shape(1)<<","<<col_buffer_xu.shape(2);

 }

  for (int g = 0; g < group_; ++g) {
	  const int M = bottom[1]->shape(0) /group_;
	  const int N = xuv_buffer_.count(2);
	  const int K = bottom[1]->count(1);
	  //const int row_offset = bottom[1]->shape(0) /group_ + 1;
    //LOG(INFO)<<"xuv: M = "<<M<<",N = "<<N<<",K = "<<K;

    //const int row_offset_ = bottom[2]->shape(0) /group_ + 1;


	  int left_cols = 0;

    //xu
		caffe_cpu_gemm<float>(CblasNoTrans, CblasNoTrans, M, N, K,
				  (float)1., weights, col_buff ,
				  (float)0., output);

  }
  float* col_buff_xuv = input_xu;
/*
  //xuv setting
    if (!is_1x1_ ||  is_concatenating_weights_features_) {
    //  if (tid == 0) im2col_time -= omp_get_wtime();
      int offset = 0;
      if (!skip_im2col || is_concatenating_weights_features_) {
        offset = bottom[2]->count(1) * xuv_buffer_.count(2)*group_*batch_idx;
        float *col_buff_mutable = col_buffer_xuv.mutable_cpu_data() + offset;
      	  conv_im2col_cpu_xuv_nc(input_xu, col_buff_mutable);

      }
      col_buff_xuv = col_buffer_xuv.mutable_cpu_data() + offset;
    //  LOG(INFO)<<"xuv col_buffer_ size: "<<col_buffer_xuv.shape(0)<<","<<col_buffer_xuv.shape(1)<<","<<col_buffer_xuv.shape(2);
      if (tid == 0) im2col_time += omp_get_wtime();
    }
*/

    for (int g = 0; g < group_; ++g) {
      const int M_ = bottom[2]->shape(0) /group_;
      const int N_ = xuv_buffer_.count(2);
      const int K_ = bottom[2]->count(1);
      //const int row_offset_ = bottom[2]->shape(0) /group_ + 1;
  	  int left_cols = 0;

      //xuv
      caffe_cpu_gemm<float>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
  				  (float)1., weights2 , col_buff_xuv,
  				  (float)0., output2);
  		//LOG(INFO)<<"caffe_cpu_gemm time: "<<omp_get_wtime() - k;

    }


}



template <>
void BaseConvolutionLayer<double>::forward_cpu_gemm_ccnmm_merge(const double* input,
    const double* weights, double* output, int batch_idx, const vector<Blob<double>*>& bottom,
    bool skip_im2col) {
  NOT_IMPLEMENTED;
}

template<>
void BaseConvolutionLayer<float>::forward_cpu_gemm_ccnmm_merge(const float* input,
    const float* weights, float* output, int batch_idx, const vector<Blob<float>*>& bottom,
    bool skip_im2col)
{
  const float* col_buff = input;
  //initial STEP
  const int M = this->blobs_[0]->shape(0);
  const int N = this->blobs_[0]->count(1,4);
  const int weight_offset = this->blobs_[0]->count();
  const int row_offset = this->blobs_[0]->shape(0) + 1;
  int masked_col_num = 0;
	int left_cols = 0;
	float group_sparsity = 0;
//  LOG(INFO)<<"initial_check"<<initial_check;

  if(initial_check == 1)
  {
LOG(INFO)<<"****inital_check****";
  squeezed_weight_buffer_ccnmm.Reshape(this->blobs_[0]->shape(0),this->blobs_[0]->shape(1),this->blobs_[0]->shape(2),this->blobs_[0]->shape(3));
//  LOG(INFO)<<"Shape of bottom"<<this->blobs_[0]->shape(0)<<','<<this->blobs_[0]->shape(1)<<','<<this->blobs_[0]->shape(2)<<','<<this->blobs_[0]->shape(3);
  //LOG(INFO)<<"ConvolutionParameter_ConvMode_LOWERED_CCNMM";
			for (int g = 0; g < group_; ++g) {
				caffe_cpu_if_all_zero(M,
						N,
						this->blobs_[0]->cpu_data() + weight_offset * g,
						col_buf_mask_ccnmm.mutable_cpu_data() + N * g);
			}
			masked_col_num = 0;
    //  LOG(INFO)<<"col_buf_mask_ccnmm.count():"<<col_buf_mask_ccnmm.count();
			for(int idx=0; idx<col_buf_mask_ccnmm.count();++idx){
      //  LOG(INFO)<<"col_buf_mask_.cpu_data()[idx]:"<<col_buf_mask_.cpu_data()[idx];
				if(col_buf_mask_ccnmm.cpu_data()[idx]){
					masked_col_num++;
				}
			}
			group_sparsity = (float)masked_col_num/(float)col_buf_mask_ccnmm.count();
		LOG(INFO) << Layer<float>::layer_param().name() << " column sparsity: " << group_sparsity;
			is_concatenating_weights_features_ = true;

      //LOG(INFO)<<"is_concatenating_weights_features_"<<is_concatenating_weights_features_;
			// compress weight matrix
      //TODO:weight_offset kernel_dim have to change later
			left_cols = 0;


			for (int g = 0; g < group_; ++g) {
        //LOG(INFO)<<"checking del_zero: before";
				caffe_cpu_del_zero_cols( this->blobs_[0]->shape(0) /group_,
					  this->blobs_[0]->count(1),
					 this->blobs_[0]->cpu_data() + weight_offset * g,
					  squeezed_weight_buffer_ccnmm.mutable_cpu_data() + weight_offset * g,
					  &left_cols,
					  col_buf_mask_ccnmm.cpu_data() + kernel_dim_ * g );
				left_columns_.push_back(left_cols);
        LOG(INFO)<<"left_colums_"<<left_cols;

			}


      initial_check = 0;
    }

        /*@@@@@@@@@@@@@@@@@@@@@
        col_buffer_shape_.clear();
      //    LOG(INFO)<<"check point 1";
        col_buffer_shape_.push_back(this->blobs_[0]->count(1) * group_ * this->blobs_[0]->count(0,1)); //col_buffer_shape_.push_back(kernel_dim_ * group_ * num_);
      //    LOG(INFO)<<"check point 2";
        col_buffer_shape_.push_back(top[1]->shape(2));//output_h
      //    LOG(INFO)<<"check point 3";
        col_buffer_shape_.push_back(top[1]->shape(3));//output_w
      //  LOG(INFO)<<"check point 4";
        col_buffer_.Reshape(col_buffer_shape_);
    //      LOG(INFO)<<"check point 5";


    //    col_buffer_.cpu_data();///???????????????
*/
//offset = col_offset_*group_*batch_idx, col_offset_ = kernel_dim_ * conv_out_spatial_dim_

        conv_im2col_cpu(input, col_buffer_ccnmm.mutable_cpu_data());
        int offset = 0;
        offset = this->blobs_[0]->count(1) * xs_buffer_.count(2) *group_*batch_idx; // col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
    //    LOG(INFO)<<"batch_idx: "<<batch_idx<<"offset: "<<offset;
        float *col_buff_mutable = /*col_buffer_ccnmm.*/col_buffer_.mutable_cpu_data() + offset;

        if(is_concatenating_weights_features_){
        //  LOG(INFO)<<"kernel_dim: "<<this->blobs_[0]->count(1)<<" conv_out_spatial_dim_: "<<top[1]->count(2);
        //  LOG(INFO)<<"group_:"<<group_<<" batch_idx"<<batch_idx;
          conv_im2col_cpu(input, col_buff_mutable, col_buf_mask_ccnmm.mutable_cpu_data()/*, dense_feature_map_mask_.mutable_cpu_data()*/);
        }

        col_buff = /*col_buffer_ccnmm.*/col_buffer_.cpu_data() + offset;

      //===========================calculate=====================================//


    /*
      parameter info:
      this->blobs_[0]->count(1) -> kernel_dim_
      top[1]->count(2) -> conv_out_spatial_dim_
      conv_out_channels_ = num_output_;
    */
      int offset_sum = 0;
      for (int g = 0; g < group_; ++g) {
          int left_cols = 0;
          left_cols = left_columns_[g];
        //  LOG(INFO)<<left_cols;
        //  LOG(INFO)<<"after start,before gemm";

        caffe_cpu_cblas_gemm(xs_buffer_.shape(1) /
  				  group_, xs_buffer_.count(2), left_cols,
  				  (float)1., squeezed_weight_buffer_ccnmm.cpu_data() + weight_offset_ * g,
  				  this->blobs_[0]->count(1) , col_buff + offset_sum,
  				  xs_buffer_.count(2), (float)0., output + output_offset_ * g, xs_buffer_.count(2));
  		  offset_sum += left_cols * xs_buffer_.count(2);
      //  LOG(INFO)<<"after gemm";
      }



}




template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    }
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_cpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_cpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
      input, bias_multiplier_.cpu_data(), 1., bias);
}

#ifndef CPU_ONLY

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    }
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_gpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_gpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
      input, bias_multiplier_.gpu_data(), 1., bias);
}

#endif  // !CPU_ONLY

INSTANTIATE_CLASS(BaseConvolutionLayer);

}  // namespace caffe
