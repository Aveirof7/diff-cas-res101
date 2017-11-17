// ----------------------------------------------------------------------------------------
// These codes were written by Xuepeng Shi, a master supervised by Prof. Shiguang Shan
// If you have any question, please contact with Xuepeng Shi at xuepeng.shi@vipl.ict.ac.cn
// Note: the above information must be kept whenever or wherever the codes are used
// ----------------------------------------------------------------------------------------

#include <cfloat>

#include "caffe/layers/roi_pooling_st_layer.hpp"


namespace caffe {

template <typename Dtype>
  __global__ void ROIPoolingStForward(const int nthreads, 
    const Dtype* bottom_data, const Dtype* source_xy, const Dtype* img_id_blob, 
      const int channels, const int src_height, const int src_width, 
      const int dst_height, const int dst_width, Dtype* top_data) {
      const int HW = dst_height * dst_width;
      const int CHW = channels * HW;
      int img_id = int(*img_id_blob);
      CUDA_KERNEL_LOOP(index, nthreads) {
        const int s_idx = index / CHW * HW + index % HW;
        const Dtype sx = ((source_xy[s_idx << 1] + 1) * src_width - 1.0) / 2.0;
        const Dtype sy = ((source_xy[s_idx << 1 | 1] + 1) * src_height - 1.0) / 2.0;
        int y = floor(sy);
        int x = floor(sx);
        int b_idx;
        if (x >= 0 && x < src_width && y >= 0 && y < src_height) {
          b_idx = index / HW * src_height * src_width + y * src_width + x + img_id * channels * src_height * src_width;
          top_data[index] += bottom_data[b_idx] * (1 - fabs(sx - x))
                          * (1 - fabs(sy - y));
        }
        x ++ ;
        if (x >= 0 && x < src_width && y >= 0 && y < src_height) {
          b_idx = index / HW * src_height * src_width + y * src_width + x + img_id * channels * src_height * src_width;
          top_data[index] += bottom_data[b_idx] * (1 - fabs(sx - x))
                          * (1 - fabs(sy - y));
        }
        x --, y ++ ;
        if (x >= 0 && x < src_width && y >= 0 && y < src_height) {
          b_idx = index / HW * src_height * src_width + y * src_width + x + img_id * channels * src_height * src_width;
          top_data[index] += bottom_data[b_idx] * (1 - fabs(sx - x))
                          * (1 - fabs(sy - y));
        }
        x ++ ;
        if (x >= 0 && x < src_width && y >= 0 && y < src_height) {
          b_idx = index / HW * src_height * src_width + y * src_width + x + img_id * channels * src_height * src_width;
          top_data[index] += bottom_data[b_idx] * (1 - fabs(sx - x))
                          * (1 - fabs(sy - y));
        }
      }
    }

template <typename Dtype>
  __global__ void ROIPoolingStSetTheta(const int nthreads,
    Dtype* theta, Dtype* bottom_rois, Dtype spatial_scale_,
    const int src_width_, const int src_height_, const int transformer_size_)
  {
    Dtype x1 = bottom_rois[1] * spatial_scale_;
    Dtype y1 = bottom_rois[2] * spatial_scale_;
    Dtype x2 = bottom_rois[3] * spatial_scale_;
    Dtype y2 = bottom_rois[4] * spatial_scale_;
    CUDA_KERNEL_LOOP(index, nthreads)
    {
      Dtype *tmp = theta + (transformer_size_ + 1) * index;
    
      *tmp = (x2 - x1 + 1) / src_width_;
      *(tmp + 1) = 0;
      *(tmp + 2) = (x2 + x1 + 1) / src_width_ - 1.0;
      *(tmp + 3) = 0;
      *(tmp + 4) = (y2 - y1 + 1) / src_height_;
      *(tmp + 5) = (y2 + y1 + 1) / src_height_ - 1.0;
    }
  }

template <typename Dtype>
  __global__ void ROIPoolingStSetImgId(const int nthreads,
    Dtype* img_id, Dtype* bottom_rois)
  {
    CUDA_KERNEL_LOOP(index, nthreads)
    {
      *img_id = bottom_rois[0];
    }
  }

template <typename Dtype>
void ROIPoolingStLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* bottom_rois = bottom[1]->mutable_gpu_data();  

  // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
  for (int r = 0; r < num_rois_; ++r) {
    Blob<Dtype> theta_;
    theta_.Reshape(1, transformer_size_ + 1, 1, 1);
    caffe_gpu_set<Dtype>(theta_.count(), Dtype(1), theta_.mutable_gpu_data());
    
    ROIPoolingStSetTheta<Dtype><<<CAFFE_GET_BLOCKS(1), CAFFE_CUDA_NUM_THREADS>>>(
      1, theta_.mutable_gpu_data(), bottom_rois, spatial_scale_, 
      src_width_, src_height_, transformer_size_);
    
    const int count = top[0]->count() / num_rois_;
    Blob<Dtype> tmp_theta_;
    tmp_theta_.Reshape(1, transformer_size_, 1, 1);
    Dtype* theta = tmp_theta_.mutable_gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data() + r * channels_ * dst_height_ * dst_width_;
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* source_xy = source_xy_.mutable_gpu_data() + r * 2 * dst_height_ * dst_width_;

    int dst_offset = dst_height_ * dst_width_;  
    caffe_gpu_set<Dtype>(count, Dtype(0), top_data);
    caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, 1, transformer_size_, 
                   transformer_size_ + 1, Dtype(1), theta_.gpu_data(), 
                   T_.gpu_data(), Dtype(0), theta);


    caffe_gpu_gemm(CblasNoTrans, CblasTrans, dst_offset, 2, 3, Dtype(1), 
                   target_xy_.gpu_data(), theta, Dtype(0), 
                   source_xy);

    Blob<Dtype> img_id;
    img_id.Reshape(1, 1, 1, 1);
    ROIPoolingStSetImgId<Dtype><<<CAFFE_GET_BLOCKS(1), CAFFE_CUDA_NUM_THREADS>>>(
      1, img_id.mutable_gpu_data(), bottom_rois);
    ROIPoolingStForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, source_xy, img_id.mutable_gpu_data(), channels_, src_height_, src_width_, 
      dst_height_, dst_width_, top_data);
    
    // Increment ROI data pointer
    bottom_rois += 5;
  }
}


template <typename Dtype>
void ROIPoolingStLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_LAYER_GPU_FUNCS(ROIPoolingStLayer);

}  // namespace caffe
