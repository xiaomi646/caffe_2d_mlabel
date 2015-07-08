clc;
clear all
 
addpath(genpath('/home/wzhang/commonly_used_function/liblinear-1.96'));
addpath('/home/wzhang/commonly_used_function');

trainX = [];
testX = [];

for i = 1 : 200
    i
    filename = strcat('/home/wzhang/caffe_2d_mlabel/vt/dmatrix/Train_feature_matrix_small_column_', num2str(i-1),'_after_Ttest.mat'); 
    load( filename );
    
    filename = strcat('/home/wzhang/caffe_2d_mlabel/vt/dmatrix/Test_feature_matrix_small_column_', num2str(i-1),'_after_Ttest.mat'); 
    load( filename );
    
    trainX = [trainX  trainX_temp];
    testX  = [testX   testX_temp];
    
                    
end
                
load('/home/wzhang/caffe_2d_mlabel/data/train_label.mat');
trainY = label;
clear label

load('/home/wzhang/caffe_2d_mlabel/data/val_label.mat');
testY = label;       
 
      
            
trainX  = ScaleRowData(trainX);
testX   = ScaleRowData(testX);  

[predicted_lable,  accuracy_value] = SVM_computing_new( trainX, trainY, testX, testY); 
accuracy_value
               
 
 

