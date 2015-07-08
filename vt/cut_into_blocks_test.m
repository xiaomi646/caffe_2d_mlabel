clc
clear all


root_dir = '/home/wzhang/caffe_2d_mlabel/vt/Prediction';

pkg load mpi;
MPI_Init();
save_default_options('-7');

CW = MPI_Comm_Load("NEWORLD");
my_rank = MPI_Comm_rank (CW);
p = MPI_Comm_size (CW);

range = floor( 200/p);
stepping_2 = 0:p;
stepping_2 = stepping_2 * range;
stepping_2(end) = 200 + 1;
stepping_2(1) = 1;

startR = stepping_2(my_rank+1);
lastR = stepping_2(my_rank+2)-1;

 
for index = startR : lastR

    filename = strcat(root_dir, '/Caffe_Matrix_Test_conv5_3_vt_models_iter_15000_',num2str(index -1 ),'.mat') 

    load(filename);
    feature_matrix = feat_label.feat;
    clear feat_label.feat;


    num_blocks = 200;

    range = floor( size(feature_matrix,2)/num_blocks); 
    stepping = 0: num_blocks;
    stepping = stepping * (range);
    stepping(end) =  size(feature_matrix,2) + 1;
    stepping(1) = 1;


    for j = 1 : num_blocks

        startR_2 = stepping( j );
        lastR_2 = stepping( j + 1)-1;

        data_block = feature_matrix(:, startR_2 : lastR_2);
        filename2 = strcat('/home/wzhang/caffe_2d_mlabel/vt/dmatrix/Test_block_feature_matrix_row',num2str(index -1 ),'_column_',num2str(j-1),'.mat');

        save(filename2,'data_block');
    end
    clear feature_matrix
    %delete(filename) 
    
end




MPI_Finalize();   

    