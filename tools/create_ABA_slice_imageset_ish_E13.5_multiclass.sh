export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:/usr/local/cuda-6.5/lib64
HOME_ROOT=/home/tzeng
PROJ_ROOT=ABA/autoGenelable_multi_lables_proj
CAFFE_TOOLS=$HOME_ROOT/$PROJ_ROOT/code/caffe/build/tools

#TRAIN_DATA_ROOT=$HOME_ROOT/$PROJ_ROOT/images/train/
#VAL_DATA_ROOT=$HOME_ROOT/$PROJ_ROOT/images/val/

MDB_DIR=$HOME_ROOT/$PROJ_ROOT/data
DATA_DIR=$HOME_ROOT/$PROJ_ROOT/data

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=224
  RESIZE_WIDTH=224
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

# if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  # echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  # echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       # "where the ImageNet training data is stored."
  # exit 1
# fi

# if [ ! -d "$VAL_DATA_ROOT" ]; then
  # echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  # echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       # "where the ImageNet validation data is stored."
  # exit 1
# fi

echo "Creating train lmdb..."


#GLOG_logtostderr=1 $CAFFE_TOOLS/convert_imageset.bin \
#    $TRAIN_DATA_ROOT \
#    $DATA_DIR/train.txt \
#    $MDB_DIR/train_lmdb 1 \
#	"leveldb" \
#    $RESIZE_HEIGHT $RESIZE_WIDTH


GLOG_logtostderr=1 $CAFFE_TOOLS/convert_image_slice_set.bin \
    $DATA_DIR/train_slice_ish_E13.5_multi_class.txt \
    $MDB_DIR/train_slice_lvdb_ish_E13.5_multi_class 1 \
	"leveldb" \
    $RESIZE_HEIGHT $RESIZE_WIDTH




echo "Creating val lmdb..."


	
#GLOG_logtostderr=1 $CAFFE_TOOLS/convert_imageset.bin \
#    $VAL_DATA_ROOT \
#    $DATA_DIR/val.txt \
#    $MDB_DIR/val_lmdb 1 \
#	"leveldb" \
#    $RESIZE_HEIGHT $RESIZE_WIDTH
	
GLOG_logtostderr=1 $CAFFE_TOOLS/convert_image_slice_set.bin \
    $DATA_DIR/all_test_slice_ish_E13.5_multi_class.txt \
    $MDB_DIR/all_test_slice_lvdb_ish_E13.5_multi_class 1 \
	"leveldb" \
    $RESIZE_HEIGHT $RESIZE_WIDTH


echo "Done."