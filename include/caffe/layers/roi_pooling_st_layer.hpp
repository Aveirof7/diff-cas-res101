// ----------------------------------------------------------------------------------------
// These codes were written by Xuepeng Shi, a master supervised by Prof. Shiguang Shan
// If you have any question, please contact with Xuepeng Shi at xuepeng.shi@vipl.ict.ac.cn
// Note: the above information must be kept whenever or wherever the codes are used
// ----------------------------------------------------------------------------------------

#ifndef CAFFE_ROI_POOLING_ST_LAYERS_HPP_
#define CAFFE_ROI_POOLING_ST_LAYERS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/* ROIPoolingLayer - Region of Interest Pooling Layer
*/
template <typename Dtype>
class ROIPoolingStLayer : public Layer<Dtype> {
 public:
  explicit ROIPoolingStLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ROIPoolingSt"; }

  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int transformer_size_;
  Dtype spatial_scale_;
  int num_rois_;
  int batch_num_;
  int channels_;
  int src_height_;
  int src_width_;
  int dst_height_;
  int dst_width_;
  /** The vector that stores the source and target coordinates as a set of blobs. */
  Blob<Dtype> target_xy_;
  Blob<Dtype> source_xy_;
  Blob<Dtype> source_dxy_;
  Blob<Dtype> T_; 

};

}  // namespace caffe

#endif  // CAFFE_FAST_RCNN_LAYERS_HPP_
