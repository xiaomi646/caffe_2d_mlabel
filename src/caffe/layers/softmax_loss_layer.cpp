// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, &softmax_top_vec_);
  if (top->size() >= 1) {
    // softmax loss (averaged across batch)
    (*top)[0]->Reshape(1, 1, 1, 1);
  }
  if (top->size() == 2) {
    // softmax output
    (*top)[1]->Reshape(bottom[0]->num(), bottom[0]->channels(),
        bottom[0]->height(), bottom[0]->width());
  }
}

template <typename Dtype>
Dtype SoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // The forward pass computes the softmax prob values.
  softmax_bottom_vec_[0] = bottom[0];
  softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int num = prob_.num();
  int dim = prob_.count() / num;
  Dtype loss = 0;
  //LOG(INFO)<< "softmax loss layer prob num ="<<num<< " Dim ="<<dim<<  "  count ="<<prob_.count() <<"  FLT_MIN ="<<FLT_MIN //<<"  FLT_MAX ="<< FLT_MAX;
  for (int i = 0; i < num; ++i) {
   
    int m_label=static_cast<int>(label[i]); // current data's multi class is represent from 0-n , where 0 means such labe
	//is not exist, and 1-n represents the #class n-1, Therefore, to comptibale with softmax which count class from 0:n-1
	// the input label value is minused by 1.
	//if (m_label>=0){  //for multilabel case, the code is modified by Tao Zeng 12/17/2014. we only count loss on the label    that is greater than /equal to zero . 
		
		loss += -log(max(prob_data[i * dim + static_cast<int>(label[i])],
						 Dtype(FLT_MIN)));
		//			 }
		
		//loss += -log(max(prob_data[i * dim + m_label],
		//				 Dtype(FLT_MIN)));
		//			 }
	//}
  }
  if (top->size() >= 1) {
    (*top)[0]->mutable_cpu_data()[0] = loss / num;
  }
  if (top->size() == 2) {
    (*top)[1]->ShareData(prob_);
  }
  return loss / num;
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = (*bottom)[1]->cpu_data();
    int num = prob_.num();
    int dim = prob_.count() / num;
    for (int i = 0; i < num; ++i) {
      bottom_diff[i * dim + static_cast<int>(label[i])] -= 1;
    }
    // Scale down gradient
    caffe_scal(prob_.count(), Dtype(1) / num, bottom_diff);
  }
}


INSTANTIATE_CLASS(SoftmaxWithLossLayer);


}  // namespace caffe
