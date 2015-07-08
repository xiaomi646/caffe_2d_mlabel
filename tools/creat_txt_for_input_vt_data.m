if ispc
    n_image_path='Z:\osi_images_datasets\gss_images_dataset\images';
    p_image_path='Z:\osi_images_datasets\gsr_images_dataset\images';

    txt_out_dir ='Z:\caffe_2d_mlabel\data';
else
    n_image_path='/home/tzeng/osi_images_datasets/gss_images_dataset/images';
    p_image_path='/home/tzeng/osi_images_datasets/gsr_images_dataset/images';
    txt_out_dir ='/home/tzeng/caffe_2d_mlabel/data';  
    
end
train_file=[txt_out_dir filesep 'train_file.txt'];
val_file  =[txt_out_dir filesep 'val_file.txt'];




N_file_strut =dir(n_image_path);
P_file_strut =dir(p_image_path);

N_files={N_file_strut.name};
N_files=N_files(3:end);
length(N_files)
for i=1:length(N_files)
    %if  ~strcmp(N_files{i}(end-3:end),'.jpg')
  %    N_files{i}=[N_files{i} ''
   % //end
    N_files{i}=[n_image_path filesep N_files{i} ' ' num2str(0)];
end

P_files={P_file_strut.name};
P_files=P_files(3:end);


for i=1:length(P_files)
    P_files{i}=[p_image_path filesep P_files{i} ' ' num2str(1)];
end

disp('done path');
all_files=[N_files P_files];

% n_labels =ones(1,length(N_files))*-1;
% p_labels =ones(1,length(P_files));
% 
% total_labels =[n_labels p_labels];

per_order = randperm(length(all_files));

train_idex =per_order(1:floor(length(all_files)*0.7));
val_idex =per_order(length(train_idex)+1:length(all_files));


train_files=all_files(train_idex);
% train_labels=total_labels(train_idx);

val_files=all_files(val_idex);
% val_labels=total_labels(val_idx);

fileID = fopen(train_file,'w');


for j=1:length(train_files)
    outline = train_files{j};
    fprintf(fileID,'%s\n',outline);
end
disp('done output train.txt');
fclose(fileID);



fileID = fopen(val_file,'w');

for j=1:length(val_files)
    outline =val_files{j};
    fprintf(fileID,'%s\n',outline);
end
disp('done output val.txt');
fclose(fileID);


