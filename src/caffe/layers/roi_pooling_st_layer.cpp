// ----------------------------------------------------------------------------------------
// These codes were written by Xuepeng Shi, a master supervised by Prof. Shiguang Shan
// If you have any question, please contact with Xuepeng Shi at xuepeng.shi@vipl.ict.ac.cn
// Note: the above information must be kept whenever or wherever the codes are used
// ----------------------------------------------------------------------------------------

#include <cfloat>

#include "caffe/layers/roi_pooling_st_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;
using std::cout;
using std::endl;

namespace caffe {

template <typename Dtype>
void ROIPoolingStLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  ROIPoolingStParameter roi_pool_st_param = this->layer_param_.roi_pooling_st_param();
  CHECK_GT(roi_pool_st_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(roi_pool_st_param.pooled_w(), 0)
      << "pooled_w must be > 0";
  dst_height_ = roi_pool_st_param.pooled_h();
  dst_width_ = roi_pool_st_param.pooled_w();
  spatial_scale_ = roi_pool_st_param.spatial_scale();
  LOG(INFO) << "Spatial scale: " << spatial_scale_;
  
}

template <typename Dtype>
void ROIPoolingStLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_rois_ = bottom[1]->num();
  batch_num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  src_height_ = bottom[0]->height();
  src_width_ = bottom[0]->width();

  transformer_size_ = 6;

  target_xy_.Reshape(1, 3, dst_height_, dst_width_);
  source_xy_.Reshape(num_rois_, 2 , dst_height_, dst_width_);
  source_dxy_.Reshape(num_rois_, 2 , dst_height_, dst_width_);

  Dtype* corr = target_xy_.mutable_cpu_data();
  for (int r = 0; r < dst_height_; ++ r) {
    for (int c = 0; c < dst_width_; ++ c) {
      corr[0] = (c * 2.0 + 1.0) / dst_width_ - 1.0;
      corr[1] = (r * 2.0 + 1.0) / dst_height_ - 1.0;
      corr[2] = Dtype(1.0); 
      corr += 3;
    }
  }

  T_.Reshape(transformer_size_ + 1, transformer_size_, 1, 1);
  Dtype* T = T_.mutable_cpu_data();
  caffe_set<Dtype>(T_.count(), Dtype(0), T);

  for (int i = 0; i < transformer_size_; ++ i) {
    T[i * transformer_size_ + i] = Dtype(1);
  }
  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  top[0]->Reshape(num_rois_, channels_, dst_height_, dst_width_);
}

template <typename Dtype>
void ROIPoolingStLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_rois = bottom[1]->cpu_data();  

  // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
  for (int r = 0; r < num_rois_; ++r) {
    int img_id = int(bottom_rois[0]);
    Dtype x1 = bottom_rois[1] * spatial_scale_;
    Dtype y1 = bottom_rois[2] * spatial_scale_;
    Dtype x2 = bottom_rois[3] * spatial_scale_;
    Dtype y2 = bottom_rois[4] * spatial_scale_;
    Blob<Dtype> theta_;
    theta_.Reshape(1, transformer_size_ + 1, 1, 1);
    caffe_set<Dtype>(theta_.count(), Dtype(1), theta_.mutable_cpu_data());

    Dtype *tmp = theta_.mutable_cpu_data();
    *tmp = (x2 - x1 + 1) / src_width_;
    *(tmp + 1) = 0;
    *(tmp + 2) = (x2 + x1 + 1) / src_width_ - 1.0;
    *(tmp + 3) = 0;
    *(tmp + 4) = (y2 - y1 + 1) / src_height_;
    *(tmp + 5) = (y2 + y1 + 1) / src_height_ - 1.0;

    Blob<Dtype> tmp_theta_;
    tmp_theta_.Reshape(1, transformer_size_, 1, 1);
    Dtype* theta = tmp_theta_.mutable_cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data() + top[0]->offset(r);
    Dtype* source_xy = source_xy_.mutable_cpu_data() + source_xy_.offset(r);
    int dst_offset = dst_height_ * dst_width_;
    caffe_set<Dtype>(top[0]->count() / num_rois_, Dtype(0), top_data);
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 1, transformer_size_, 
                   transformer_size_ + 1, Dtype(1), theta_.cpu_data(),
                   T_.cpu_data(), Dtype(0), theta); 

    caffe_cpu_gemm(CblasNoTrans, CblasTrans, dst_offset, 2, 3, Dtype(1), 
                     target_xy_.cpu_data(), theta, Dtype(0), source_xy);
    for (int h = 0; h < dst_height_; ++ h) {
      for (int w = 0; w < dst_width_; ++ w) {
        Dtype sx = ((source_xy[0] + 1.0) * src_width_ - 1.0) / 2.0;
        Dtype sy = ((source_xy[1] + 1.0) * src_height_ - 1.0) / 2.0;
        for (int c = 0; c < channels_; ++ c) {
          for (int x = floor(sx); x <= ceil(sx); ++ x) {
            for (int y = floor(sy); y <= ceil(sy); ++ y) {
              if (x >= 0 && y >= 0 && x < src_width_ && y < src_height_)
                top_data[top[0]->offset(0, c, h, w)] += 
                bottom[0]->data_at(img_id, c, y, x) * (1 - fabs(sx - x)) * (1 - fabs(sy - y));                    
            }
          }
        }
        source_xy += 2;
      }
    }
    
    // Increment ROI data pointer
    bottom_rois += bottom[1]->offset(1);
  }

}

template <typename Dtype>
void ROIPoolingStLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
}


#ifdef CPU_ONLY
STUB_GPU(ROIPoolingStLayer);
#endif

INSTANTIATE_CLASS(ROIPoolingStLayer);
REGISTER_LAYER_CLASS(ROIPoolingSt);

}  // namespace caffe
