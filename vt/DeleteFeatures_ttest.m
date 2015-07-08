clc;
clear all
addpath('/home/wzhang/commonly_used_function');

pkg load mpi;
MPI_Init();
save_default_options('-7');

CW = MPI_Comm_Load("NEWORLD");
my_rank = MPI_Comm_rank (CW);
p = MPI_Comm_size (CW);
 
num_blocks = 200;
 
range = floor( 200/p);
stepping_2 = 0:p;
stepping_2 = stepping_2 * range;
stepping_2(end) = 200 + 1;
stepping_2(1) = 1;

startR = stepping_2(my_rank+1);
lastR = stepping_2(my_rank+2)-1;
    
trainX = [];
testX  = [];   
for j =  startR : lastR 
 
    filename = strcat('/home/wzhang/caffe_2d_mlabel/vt/dmatrix/Train_feature_matrix_small_column_', num2str(j-1),'.mat');
    load(filename);             
    trainX_temp = feature_matrix_small;


    filename = strcat('/home/wzhang/caffe_2d_mlabel/vt/dmatrix/Test_feature_matrix_small_column_', num2str(j-1),'.mat');
    load(filename);             
    testX_temp = feature_matrix_small;

    clear feature_matrix_small;   


    clear idx sum_temp
    sum_temp = sum(trainX_temp);
    idx = find(sum_temp ~= 0);
    trainX_temp = trainX_temp(:, idx);
    testX_temp  = testX_temp (:, idx);
    clear idx

    load('/home/wzhang/caffe_2d_mlabel/data/train_label.mat');
    trainY = label;
    clear label
    load('/home/wzhang/caffe_2d_mlabel/data/val_label.mat');
    testY = label;    
                    
    pvalues  = fsTtest(trainX_temp, trainY);
    
    if size(pvalues,2) == 1 && size(pvalues,1) > 1
         pvalues = pvalues';
    end

    idx_ttest = find(pvalues < 0.08 & pvalues > 0);
    trainX_temp = trainX_temp(:, idx_ttest);
    testX_temp  = testX_temp (:, idx_ttest);    
    size(trainX_temp,2)
    size(testX_temp,2)
    filename = strcat('/home/wzhang/caffe_2d_mlabel/vt/dmatrix/Train_feature_matrix_small_column_', num2str(j-1),'_after_Ttest.mat');
    save(filename,'trainX_temp');             

    filename = strcat('/home/wzhang/caffe_2d_mlabel/vt/dmatrix/Test_feature_matrix_small_column_', num2str(j-1),'_after_Ttest.mat');
    save(filename, 'testX_temp');            
end

 

MPI_Finalize();   

