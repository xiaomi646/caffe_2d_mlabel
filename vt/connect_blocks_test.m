clc
clear all

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

 
    feature_matrix_small = [];

    num_blocks = 200;

    for j = 1 : num_blocks

        filename2 = strcat('/home/wzhang/caffe_2d_mlabel/vt/dmatrix/Test_block_feature_matrix_row',num2str(j-1),'_column_',num2str(index-1),'.mat');
        load(filename2);
        feature_matrix_small = [feature_matrix_small; data_block];
        
    end

    filename = strcat('/home/wzhang/caffe_2d_mlabel/vt/dmatrix/Test_feature_matrix_small_column_', num2str(index -1 ),'.mat');
     
    save(filename, 'feature_matrix_small');

    for j = 1 : num_blocks

        filename2 = strcat('/home/wzhang/caffe_2d_mlabel/vt/dmatrix/Test_block_feature_matrix_row',num2str(j-1),'_column_',num2str(index -1),'.mat');
        delete(filename2);
       
    end
end
MPI_Finalize();   

    