clc
clear all
close all


[~,b]= system('hostname');
if strcmp(b(1:4),'hpcd') == 1
    hostname_file = '/home/wzhang/caffe_2d_mlabel/vt/hostname ';
end
hostname_file
mkdir('/home/wzhang/caffe_2d_mlabel/vt/dmatrix');
% system(['mpirun -np ' num2str(25) ' -machinefile '  hostname_file   'octave cut_into_blocks_test.m']);
% system(['mpirun -np ' num2str(25) ' -machinefile '  hostname_file   'octave cut_into_blocks_train.m']);

% system(['mpirun -np ' num2str(25) ' -machinefile '  hostname_file   'octave connect_blocks_test.m']);
% system(['mpirun -np ' num2str(25) ' -machinefile '  hostname_file   'octave connect_blocks_train.m']);

system(['mpirun -np ' num2str(25) ' -machinefile '  hostname_file   'octave DeleteFeatures_ttest.m']);

% Step2_build_classifier_parallel_MPI_block_accurate

 
 