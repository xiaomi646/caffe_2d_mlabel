#include <vector>

//#include "caffe/common_layers.hpp"
//#include "caffe/layer.hpp"
//#include "caffe/util/math_functions.hpp"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
namespace caffe {

template <typename Dtype>
void SilenceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  for (int i = 0; i < bottom->size(); ++i) {
    if (propagate_down[i]) {
      caffe_set((*bottom)[i]->count(), Dtype(0),
                (*bottom)[i]->mutable_cpu_data());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SilenceLayer);
#endif

INSTANTIATE_CLASS(SilenceLayer);

}  // namespace caffe
