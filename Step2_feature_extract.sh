#!/bin/bash
export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:/usr/local/cuda-6.5/lib64
codehome=/home/wzhang/caffe_2d_mlabel/matlab
caffe_tool_dir=/home/wzhang/caffe_2d_mlabel/build/tools
pycodePath=/home/wzhang/caffe_2d_mlabel/tools
cd $codehome
DeviceID=7
# set which steps to run ----------------
run_extract_im_feature=true
run_convert_LVDB_to_MAT=true
run_svm=false
#------------------------------------------
#------------------------------------------
#------------------------------------------
layer_array=(conv5_3)


mf_name=vt_models_iter_15000

#deploy_file=deploy.prototxt

#------------------------------------------
#------------------------------------------
#------------------------------------------

target_save_data_dir=/home/wzhang/caffe_2d_mlabel/vt/Prediction
model_dir=/home/wzhang/caffe_2d_mlabel/models
temp_dir=/home/wzhang/caffe_2d_mlabel/vt/temp
# result_dir=/home/rli/caffe_3D_FastSEG/AD/result ## This is related with SVM

cnt=${#layer_array[@]}
num_train=54521
num_test=23367



# test_feature_extract_model_def_file=$model_dir/extract_test_mri2pet_multi_labels.prototxt
train_feature_extract_model_def_file=$model_dir/extract_train.prototxt
test_feature_extract_model_def_file=$model_dir/extract_train.prototxt
model_file=/home/wzhang/caffe_2d_mlabel/models/$mf_name

db_label_train_path=$temp_dir/train_label
db_label_test_path=$temp_dir/test_label

for ((i=0;i<cnt;i++)); do
     train_feature_path_array[i]="$temp_dir/train_feature_${layer_array[i]}"
	 test_feature_path_array[i]="$temp_dir/test_feature_${layer_array[i]}"
     echo "${feature_path_array[i]}"
done

db_train_feature_paths=${train_feature_path_array[0]}
db_test_feature_paths=${test_feature_path_array[0]}

layer_string=${layer_array[0]}

for ((i=1;i<cnt;i++)); do
	 db_train_feature_paths=$db_train_feature_paths,${train_feature_path_array[i]}
	 db_test_feature_paths=$db_test_feature_paths,${test_feature_path_array[i]}
	 layer_string=$layer_string,${layer_array[i]}    
done


#1. extract feature image level feature
if [ "$run_extract_im_feature" = true ];then
  for ((i=0;i<cnt;i++)); do
    rm -rf ${train_feature_path_array[i]}
	rm -rf ${test_feature_path_array[i]}
  done
	rm -rf $db_label_train_path
	rm -rf $db_label_test_path
			
	GLOG_logtostderr=1 $caffe_tool_dir/extract_features.bin $model_file $train_feature_extract_model_def_file $layer_string,label $db_train_feature_paths,$db_label_train_path $num_train GPU $DeviceID
	GLOG_logtostderr=1 $caffe_tool_dir/extract_features.bin $model_file $test_feature_extract_model_def_file $layer_string,label $db_test_feature_paths,$db_label_test_path $num_test GPU $DeviceID
	
fi

#2. convert feature level DB to matlab file

# echo save matlab file is $mat_test_data_file
if [ "$run_convert_LVDB_to_MAT" = true ];then
    for ((i=0;i<cnt;i++)); do
	mat_train_data_file=$target_save_data_dir/Caffe_Matrix_Train'_'${layer_array[i]}'_'$mf_name.mat
    mat_test_data_file=$target_save_data_dir/Caffe_Matrix_Test'_'${layer_array[i]}'_'$mf_name.mat
	
	python $pycodePath/feature_LDB_to_mat_new.py --label_db $db_label_train_path --feature_db ${train_feature_path_array[i]} --mat_file $mat_train_data_file
	python $pycodePath/feature_LDB_to_mat_new.py --label_db $db_label_test_path --feature_db ${test_feature_path_array[i]} --mat_file $mat_test_data_file
	done
fi




# #3. train and test svm on section level features

# done
# if [ "$run_svm" = true ];then
# for ((i=0;i<cnt;i++)); do
	# temp_commd=$codehome/comd7.m
		
			# mat_train_data_file=$target_save_data_dir/Caffe_Matrix_Train'_'${layer_array[i]}'_'$mf_name.mat
			# mat_test_data_file=$target_save_data_dir/Caffe_Matrix_Test'_'${layer_array[i]}'_'$mf_name.mat
			# result_file=$result_dir/auc_report_classified'_'${layer_array[i]}'_'$mf_name.mat
			# echo "run_svm_train_on_caffe_feature('$mat_train_data_file','$mat_test_data_file','$result_file');">$temp_commd
			# echo "exit">>$temp_commd
# ssh tzeng@tesla.cs.odu.edu<<'ENDSSH'
# cd /home/wzhang/caffe_2d_mlabel/
# hostname
# matlab -r "comd7"
# #ENDSSH
# done
# fi
