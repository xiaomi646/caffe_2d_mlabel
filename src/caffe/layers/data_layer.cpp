// Copyright 2014 BVLC and contributors.

#include <stdint.h>
#include <leveldb/db.h>
#include <pthread.h>

#include <string>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"

using std::string;

namespace caffe {

template <typename Dtype>
void* DataLayerPrefetch(void* layer_pointer) {
  CHECK(layer_pointer);
  DataLayer<Dtype>* layer = static_cast<DataLayer<Dtype>*>(layer_pointer);
  CHECK(layer);
  Datum datum;
  CHECK(layer->prefetch_data_);
  Dtype* top_data = layer->prefetch_data_->mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  int num_labels = 0;
  if (layer->output_labels_) {
    top_label = layer->prefetch_label_->mutable_cpu_data();
    num_labels = layer->prefetch_label_->channels();
  }
  const Dtype scale = layer->layer_param_.data_param().scale();
  const int batch_size = layer->layer_param_.data_param().batch_size();
  const int crop_size = layer->layer_param_.data_param().crop_size();
  const bool mirror = layer->layer_param_.data_param().mirror();

  if (mirror && crop_size == 0) {
    LOG(FATAL) << "Current implementation requires mirror and crop_size to be "
        << "set at the same time.";
  }
  // datum scales
  const int channels = layer->datum_channels_;
  const int height = layer->datum_height_;
  const int width = layer->datum_width_;
  const int size = layer->datum_size_;
  const Dtype* mean = layer->data_mean_.cpu_data();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    bool accept_label =false;
    do{
          switch (layer->layer_param_.data_param().backend()) {
          case DataParameter_DB_LEVELDB:
              layer->iter_->Next();
              if (!layer->iter_->Valid()) {
                // We have reached the end. Restart from the first.
                DLOG(INFO) << "Restarting data prefetching from start.";
                layer->iter_->SeekToFirst();
              }
              CHECK(layer->iter_);
              CHECK(layer->iter_->Valid());
              datum.ParseFromString(layer->iter_->value().ToString());
            break;
          case DataParameter_DB_LMDB:
            if (mdb_cursor_get(layer->mdb_cursor_, &layer->mdb_key_,
                    &layer->mdb_value_, MDB_NEXT) != MDB_SUCCESS) {
              // We have reached the end. Restart from the first.
              DLOG(INFO) << "Restarting data prefetching from start.";
              CHECK_EQ(mdb_cursor_get(layer->mdb_cursor_, &layer->mdb_key_,
                      &layer->mdb_value_, MDB_FIRST), MDB_SUCCESS);
            }
            CHECK_EQ(mdb_cursor_get(layer->mdb_cursor_, &layer->mdb_key_,
                    &layer->mdb_value_, MDB_GET_CURRENT), MDB_SUCCESS);
            datum.ParseFromArray(layer->mdb_value_.mv_data,
                layer->mdb_value_.mv_size);

            break;
          default:
            LOG(FATAL) << "Unknown database backend";
          }

          if (layer->output_labels_) {
            CHECK_EQ(datum.label_size(), num_labels);
            for (int l = 0; l < num_labels; ++l) {
              top_label[item_id * num_labels + l] = datum.label(l);
            }
            if(layer->balancing_label_&&num_labels ==1){
              Dtype label  = top_label[item_id ];
              accept_label = layer->accept_given_label(label);
              if(accept_label)
                {	top_label[item_id] = layer->get_converted_label(label);}
              }else{
               accept_label = true;}
          }


          // switch (layer->layer_param_.data_param().backend()) {
          // case DataParameter_DB_LEVELDB:
          //   CHECK(layer->iter_);
          //   CHECK(layer->iter_->Valid());
          //   datum.ParseFromString(layer->iter_->value().ToString());
          //   break;
          // case DataParameter_DB_LMDB:
          //   CHECK_EQ(mdb_cursor_get(layer->mdb_cursor_, &layer->mdb_key_,
          //           &layer->mdb_value_, MDB_GET_CURRENT), MDB_SUCCESS);
          //   datum.ParseFromArray(layer->mdb_value_.mv_data,
          //       layer->mdb_value_.mv_size);
          //   break;
          // default:
          //   LOG(FATAL) << "Unknown database backend";
          // }

      }while(!accept_label && layer->output_labels_&&num_labels==1);

          const string& data = datum.data();
          if (crop_size) {
            CHECK(data.size()) << "Image cropping only support uint8 data";
            int h_off, w_off;
            // We only do random crop when we do training.
            if (layer->phase_ == Caffe::TRAIN) {
              h_off = layer->PrefetchRand() % (height - crop_size);
              w_off = layer->PrefetchRand() % (width - crop_size);
            } else {
              h_off = (height - crop_size) / 2;
              w_off = (width - crop_size) / 2;
            }
            if (mirror && layer->PrefetchRand() % 2) {
              // Copy mirrored version
              for (int c = 0; c < channels; ++c) {
                for (int h = 0; h < crop_size; ++h) {
                  for (int w = 0; w < crop_size; ++w) {
                    int top_index = ((item_id * channels + c) * crop_size + h)
                                    * crop_size + (crop_size - 1 - w);
                    int data_index = (c * height + h + h_off) * width + w + w_off;
                    Dtype datum_element =
                        static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
                    top_data[top_index] = (datum_element - mean[data_index]) * scale;
                  }
                }
              }
            } else {
              // Normal copy
              for (int c = 0; c < channels; ++c) {
                for (int h = 0; h < crop_size; ++h) {
                  for (int w = 0; w < crop_size; ++w) {
                    int top_index = ((item_id * channels + c) * crop_size + h)
                                    * crop_size + w;
                    int data_index = (c * height + h + h_off) * width + w + w_off;
                    Dtype datum_element =
                        static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
                    top_data[top_index] = (datum_element - mean[data_index]) * scale;
                  }
                }
              }
            }
          } else {
            // we will prefer to use data() first, and then try float_data()
            if (data.size()) {
              for (int j = 0; j < size; ++j) {
                Dtype datum_element =
                    static_cast<Dtype>(static_cast<uint8_t>(data[j]));
                top_data[item_id * size + j] = (datum_element - mean[j]) * scale;
              }
            } else {
              for (int j = 0; j < size; ++j) {
                top_data[item_id * size + j] =
                    (datum.float_data(j) - mean[j]) * scale;
              }
            }
          }
          // Copy all the labels from datum
          if (layer->output_labels_) {
            CHECK_EQ(datum.label_size(), num_labels);
            for (int l = 0; l < num_labels; ++l) {
              top_label[item_id * num_labels + l] = datum.label(l);
            }
          }
          // go to the next iter
          // switch (layer->layer_param_.data_param().backend()) {
          // case DataParameter_DB_LEVELDB:
          //   layer->iter_->Next();
          //   if (!layer->iter_->Valid()) {
          //     // We have reached the end. Restart from the first.
          //     DLOG(INFO) << "Restarting data prefetching from start.";
          //     layer->iter_->SeekToFirst();
          //   }
          //   break;
          // case DataParameter_DB_LMDB:
          //   if (mdb_cursor_get(layer->mdb_cursor_, &layer->mdb_key_,
          //           &layer->mdb_value_, MDB_NEXT) != MDB_SUCCESS) {
          //     // We have reached the end. Restart from the first.
          //     DLOG(INFO) << "Restarting data prefetching from start.";
          //     CHECK_EQ(mdb_cursor_get(layer->mdb_cursor_, &layer->mdb_key_,
          //             &layer->mdb_value_, MDB_FIRST), MDB_SUCCESS);
          //   }
          //   break;
          // default:
          //   LOG(FATAL) << "Unknown database backend";
          // }
        }

return static_cast<void*>(NULL);
}

template <typename Dtype>
DataLayer<Dtype>::~DataLayer<Dtype>() {
  JoinPrefetchThread();
  // clean up the database resources
  switch (this->layer_param_.data_param().backend()) {
  case DataParameter_DB_LEVELDB:
    break;  // do nothing
  case DataParameter_DB_LMDB:
    mdb_cursor_close(mdb_cursor_);
    mdb_close(mdb_env_, mdb_dbi_);
    mdb_txn_abort(mdb_txn_);
    mdb_env_close(mdb_env_);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }
}

template <typename Dtype>
void DataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  if (top->size() == 2) {
    output_labels_ = true;
  } else {
    output_labels_ = false;
  }
  // Initialize DB
  switch (this->layer_param_.data_param().backend()) {
  case DataParameter_DB_LEVELDB:
    {
    leveldb::DB* db_temp;
    leveldb::Options options;
    options.create_if_missing = false;
    options.max_open_files = 100;
    LOG(INFO) << "Opening leveldb " << this->layer_param_.data_param().source();
    leveldb::Status status = leveldb::DB::Open(
        options, this->layer_param_.data_param().source(), &db_temp);
    CHECK(status.ok()) << "Failed to open leveldb "
                       << this->layer_param_.data_param().source() << std::endl
                       << status.ToString();
    db_.reset(db_temp);
	// iter_ is a memeber obj  =shared_ptr<leveldb::Iterator> iter_;
    iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
    iter_->SeekToFirst();
    }
    break;
  case DataParameter_DB_LMDB:
    CHECK_EQ(mdb_env_create(&mdb_env_), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS);  // 1TB
    CHECK_EQ(mdb_env_open(mdb_env_,
             this->layer_param_.data_param().source().c_str(),
             MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS)
        << "mdb_open failed";
    CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS)
        << "mdb_cursor_open failed";
    LOG(INFO) << "Opening lmdb " << this->layer_param_.data_param().source();
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST),
        MDB_SUCCESS) << "mdb_cursor_get failed";
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }

  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
                        this->layer_param_.data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      switch (this->layer_param_.data_param().backend()) {
      case DataParameter_DB_LEVELDB:
        iter_->Next();
        if (!iter_->Valid()) {
          iter_->SeekToFirst();
        }
        break;
      case DataParameter_DB_LMDB:
        if (mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_NEXT)
            != MDB_SUCCESS) {
          CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_,
                   MDB_FIRST), MDB_SUCCESS);
        }
        break;
      default:
        LOG(FATAL) << "Unknown database backend";
      }
    }
  }
  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  switch (this->layer_param_.data_param().backend()) {
  case DataParameter_DB_LEVELDB:
    datum.ParseFromString(iter_->value().ToString());
    break;
  case DataParameter_DB_LMDB:
    datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }

  // image
  int crop_size = this->layer_param_.data_param().crop_size();
  LOG(INFO) <<"Data layer input  batch_size is = "<<this->layer_param_.data_param().batch_size();
  if (crop_size > 0) {
    (*top)[0]->Reshape(this->layer_param_.data_param().batch_size(),
                       datum.channels(), crop_size, crop_size);
	// shared_ptr<Blob<Dtype> > prefetch_data_
    prefetch_data_.reset(new Blob<Dtype>(
        this->layer_param_.data_param().batch_size(), datum.channels(),
        crop_size, crop_size));
  } else {
    (*top)[0]->Reshape(
        this->layer_param_.data_param().batch_size(), datum.channels(),
        datum.height(), datum.width());
    prefetch_data_.reset(new Blob<Dtype>(
        this->layer_param_.data_param().batch_size(), datum.channels(),
        datum.height(), datum.width()));
  }
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
  if (output_labels_) {
    CHECK_GT(datum.label_size(), 0) << "Datum should contain labels for top";
    (*top)[1]->Reshape(this->layer_param_.data_param().batch_size(),
      datum.label_size(), 1, 1);
    LOG(INFO) << "output label size: " << (*top)[1]->num() << ","
      << (*top)[1]->channels() << "," << (*top)[1]->height() << ","
      << (*top)[1]->width();
    prefetch_label_.reset(
        new Blob<Dtype>(this->layer_param_.data_param().batch_size(),
          datum.label_size(), 1, 1));
  }
  // datum size
  datum_channels_ = datum.channels();
  datum_height_ = datum.height();
  datum_width_ = datum.width();
  datum_size_ = datum.channels() * datum.height() * datum.width();
  CHECK_GT(datum_height_, crop_size);
  CHECK_GT(datum_width_, crop_size);
  // check if we want to have mean
  if (this->layer_param_.data_param().has_mean_file()) {
    const string& mean_file = this->layer_param_.data_param().mean_file();
    LOG(INFO) << "Loading mean file from" << mean_file;
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
    CHECK_EQ(data_mean_.num(), 1);
    CHECK_EQ(data_mean_.channels(), datum_channels_);
    CHECK_EQ(data_mean_.height(), datum_height_);
    CHECK_EQ(data_mean_.width(), datum_width_);
  } else {
    // Simply initialize an all-empty mean.
    data_mean_.Reshape(1, datum_channels_, datum_height_, datum_width_);
  }
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  prefetch_data_->mutable_cpu_data();
  if (output_labels_) {
    prefetch_label_->mutable_cpu_data();
  }
  data_mean_.cpu_data();
  ProcessLabelSelectParam();
  DLOG(INFO) << "Initializing prefetch";
  CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template<typename Dtype>
void DataLayer<Dtype>::ProcessLabelSelectParam(){
    //this->layer_param_
	num_labels_      =this->layer_param_.data_param().label_select_param().num_labels() ;
	balancing_label_ =this->layer_param_.data_param().label_select_param().balance();
	map2order_label_ =this->layer_param_.data_param().label_select_param().reorder_label();
	//num_top_label_balance_ =this->layer_param_.data_param().label_select_param().num_top_label_balance();
	//CHECK_EQ(is_label_blance,layer_param_.has_class_prob_mapping_file());
	//const string  label_prob_map_file = layer_param_.class_prob_mapping_file();
	bool has_prob_file=this->layer_param_.data_param().label_select_param().has_class_prob_mapping_file();
	if(balancing_label_)
	   {
		CHECK_EQ(balancing_label_,has_prob_file);
		const string&  label_prob_map_file = this->layer_param_.data_param().label_select_param().class_prob_mapping_file();
		ReadLabelProbMappingFile(label_prob_map_file);
		LOG(INFO)<<"Done ReadLabelProbMappingFile";
		compute_label_skip_rate();
		LOG(INFO)<<"compute_label_skip_rate()";
		}
}

template <typename Dtype>
void DataLayer<Dtype>::ReadLabelProbMappingFile(const string& source){
  Label_Prob_Mapping_Param lb_param;
  ReadProtoFromTextFileOrDie(source, &lb_param);
  ignore_rest_of_label_ 				= 	lb_param.ignore_rest_of_label();
  rest_of_label_mapping_ 				= 	lb_param.rest_of_label_mapping();
  rest_of_label_mapping_label_			=	lb_param.rest_of_label_mapping_label();
  rest_of_label_prob_					=	lb_param.rest_of_label_prob();
  num_labels_with_prob_  				= 	lb_param.label_prob_mapping_info_size();
  if(this->layer_param_.data_param().label_select_param().has_num_top_label_balance())
	num_top_label_balance_ =this->layer_param_.data_param().label_select_param().num_top_label_balance();
  else
	num_top_label_balance_ =  num_labels_with_prob_;

  CHECK_GE(num_top_label_balance_,1);
  CHECK_GE(num_labels_with_prob_,num_top_label_balance_);
  CHECK_GE(num_labels_,2);
  CHECK_GE(num_labels_,num_labels_with_prob_);
  LOG(INFO)<<"rest_of_label_mapping_  = "<<rest_of_label_mapping_<<" "<<rest_of_label_mapping_label_;
  label_prob_map_.clear();
  label_mapping_map_.clear();
  LOG(INFO)<< "label_prob_map_ size =" <<label_prob_map_.size();
  for (int i=0;i<num_labels_with_prob_;++i){
	const Label_Prob_Mapping&   label_prob_mapping_param = lb_param.label_prob_mapping_info(i);
	int   label 			=	label_prob_mapping_param.label();
	float lb_prob			=   label_prob_mapping_param.prob();
	int   mapped_label ;
	if(label_prob_mapping_param.has_map2label())
	    mapped_label =   label_prob_mapping_param.map2label();
	else
		 mapped_label = label ;
	label_prob_map_[label]	=	lb_prob;
	label_mapping_map_[label]=   mapped_label;

  }


}

typedef std::pair<int, float> PAIR;
struct CmpByValue {
  bool operator()(const PAIR& lhs, const PAIR& rhs)
  {return lhs.second > rhs.second;}
 };



template <typename Dtype>
void DataLayer<Dtype>::compute_label_skip_rate()
{
  //float rest_of_prob =0;
  float scale_factor =0;
  vector<PAIR> label_prob_vec(label_prob_map_.begin(), label_prob_map_.end());
  sort(label_prob_vec.begin(), label_prob_vec.end(), CmpByValue()); //prob descend order;


  float bottom_prob=label_prob_vec[num_top_label_balance_-1].second;
      //for(int i=0;i<num_top_label_balance_;i++)
       //LOG(INFO)<<"num_top_label_balance_["<< label_prob_vec[i].first<<"] = "<<label_prob_vec[i].second;
	  if(!ignore_rest_of_label_){
		  //for (int i=num_top_label_balance_;i<num_labels_with_prob_;++i)
		  //{
		//	rest_of_prob+=label_prob_vec[i].second;
		 // }
		//	rest_of_prob+=rest_of_label_prob_;
		    scale_factor =bottom_prob < rest_of_label_prob_? 1.0/bottom_prob: 1.0/rest_of_label_prob_;
	  }
	  else
	  {
		 scale_factor =1.0/bottom_prob ;

	  }
	  LOG(INFO)<<" scale_factor =  "<< scale_factor;
	  LOG(INFO)<<" bottom_prob =   " << bottom_prob;
	  LOG(INFO)<<"label_prob_vec.size = "<<label_prob_vec.size();
	 // sleep(10);

	  label_prob_map_.clear();
	  // remove the class that has prob lower that top k classes;
	  for(int i=0;i<num_top_label_balance_;++i)
	  {
	       int lb= label_prob_vec[i].first;
		   float prob =label_prob_vec[i].second;
		   label_prob_map_[lb]=prob;
		   // mapping the label based on freq
		   if(map2order_label_)
		   {
				label_mapping_map_[lb]=i;
		   }
	  }

	  // Init the rest of label class

	  for (int i=0;i<num_labels_ ;++i)
	  {
		int label =i;
		if(label_prob_map_.find(label)==label_prob_map_.end())
        {
			if(ignore_rest_of_label_){
				label_prob_map_[label]	=	0;
				}
			else
			   {
				 int rest_of_label =(num_labels_-num_top_label_balance_);
				 if(rest_of_label>0)
					label_prob_map_[label]	=	rest_of_label_prob_;///rest_of_label;
					 //LOG(INFO)<<"rest_of_label_prob["<<label<<"]=" <<label_prob_map_[label];
				}
			if(rest_of_label_mapping_){
				label_mapping_map_[label] =   rest_of_label_mapping_label_;
				//LOG(INFO)<<"rest_of_label_mapping_["<<label<<"]=" <<label_mapping_map_[label];
				}
			else
				label_mapping_map_[label] =   label;
			// if reorder label is set, override the rest_of_label_mapping_
			if(map2order_label_){
			    label_mapping_map_[label] =   num_top_label_balance_;
			}
		}
	 }
	 //sleep(20);
	  //auto iterSkipRate  =label_mapping_map_.begin();
	  std::map<int,float>::iterator iterProb;
      for (iterProb = label_prob_map_.begin(); iterProb !=label_prob_map_.end(); ++iterProb) {
				label_skip_rate_map_[iterProb->first] =ceil(iterProb->second*scale_factor);
				//LOG(INFO)<<"label_skip_rate_map_["<<iterProb->first<<"]=" <<label_skip_rate_map_[iterProb->first];


		}

}


template <typename Dtype>
bool DataLayer<Dtype>::accept_given_label(const int label)
{
		//
		//balancing_label_
		if(!balancing_label_)
		    return true;
		//LOG(INFO)<<"label_skip_rate_map_["<<label<<"] =" <<label_skip_rate_map_[label];
		if (label_skip_rate_map_[label] ==0)
		   return false;
		int reminder =PrefetchRand()%label_skip_rate_map_[label];
		if(reminder ==0)
		    return true;
		else
			return false;
}

template <typename Dtype>
int  DataLayer<Dtype>::get_converted_label(const int label){
         if(!balancing_label_)
		     return label;
	    else
			return label_mapping_map_[label];
}




template <typename Dtype>
void DataLayer<Dtype>::CreatePrefetchThread() {
  // phase_ = Caffe::phase();
  // const bool prefetch_needs_rand = (phase_ == Caffe::TRAIN) &&
  //     (this->layer_param_.data_param().mirror() ||
  //      this->layer_param_.data_param().crop_size());
  // if (prefetch_needs_rand) {
  //   const unsigned int prefetch_rng_seed = caffe_rng_rand();
  //   prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  // } else {
  //   prefetch_rng_.reset();
  // }

  const unsigned int prefetch_rng_seed = caffe_rng_rand();
  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));

  // Create the thread.
  CHECK(!pthread_create(&thread_, NULL, DataLayerPrefetch<Dtype>,
        static_cast<void*>(this))) << "Pthread execution failed.";
}

template <typename Dtype>
void DataLayer<Dtype>::JoinPrefetchThread() {
  CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
}

template <typename Dtype>
unsigned int DataLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

template <typename Dtype>
Dtype DataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // First, join the thread
  JoinPrefetchThread();
  // Copy the data
  caffe_copy(prefetch_data_->count(), prefetch_data_->cpu_data(),
             (*top)[0]->mutable_cpu_data());
  if (output_labels_) {
    caffe_copy(prefetch_label_->count(), prefetch_label_->cpu_data(),
               (*top)[1]->mutable_cpu_data());
  }
  // Start a new prefetch thread
  CreatePrefetchThread();
  return Dtype(0.);
}

INSTANTIATE_CLASS(DataLayer);

}  // namespace caffe
