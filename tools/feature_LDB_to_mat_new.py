import sys, getopt


sys.path.insert(0, '/home/wzhang/caffe_2d_mlabel/python')
#sys.path.insert(0, '/home/tzeng/ABA/autoGenelable_multi_lables_proj/code/caffe/python')
# sys.path.insert(0,'/home/tzeng/autoGenelable_multi_lables_proj/code/py-leveldb-read-only/build/lib.linux-x86_64-2.7')
import os
#import lmdb
import leveldb
#import scipy.io as sio
import numpy as np
import hdf5storage

import time
caffe_root = '/home/wzhang/caffe_2d_mlabel'
#sys.path.insert(os.path.join(caffe_root, 'python'))
import caffe.io
from caffe.proto import caffe_pb2
# db_path = '/home/rli/Downloaded_Software/caffe/examples/ISBI/foreground/train_lmdb_Consecutive_slices_right'

db_path_label ='/home/tzeng/ABA/autoGenelable_multi_lables_proj/code/caffe/models/vgg_16_fine_tune_gen/label_test'
db_path_feats ='/home/tzeng/ABA/autoGenelable_multi_lables_proj/code/caffe/models/vgg_16_fine_tune_gen/flat_conv5_1_eltmax_test'
#db_path = '/home/tzeng/ABA/autoGenelable_multi_lables_proj/data/all_test_slice_lvdb_ish'

def main(argv):
	db_path_label =''
	db_path_feats =''
	mat_file =''
	print argv
	try:
		opts, args = getopt.getopt(argv,"l:f:o",["label_db=","feature_db=","mat_file="])
	except getopt.GetoptError:
		print 'feature_LDB_to_mat.py -l <label_db> -f <feature_db> -m <output_mat_file>'
		sys.exit(2)

	print opts
	print args


	for opt, arg in opts:
		if opt in ("-l","--label_db"):
			db_path_label=arg
		elif opt in("-f","--feature_db"):
			db_path_feats=arg
		elif opt in("-o","--mat_file"):
			mat_file=arg
		print arg+" "+opt

	print(db_path_label)
	print(db_path_feats)
	print(mat_file)

	if not os.path.exists(db_path_label):
		raise Exception('db label not found')
	if not os.path.exists(db_path_feats):
		raise Exception('db feature not found')



	db_label=leveldb.LevelDB(db_path_label)
	db_feats=leveldb.LevelDB(db_path_feats)
	#window_num =686
	datum = caffe_pb2.Datum()
	datum_lb = caffe_pb2.Datum()
	start=time.time();
	#ft = np.zeros((window_num, float(81)))
	#ft = np.zeros((window_num, float(100352)))
	#lb = np.zeros((window_num, float(81)))
	is_float_data =True
	window_num=0
	for key in db_feats.RangeIter(include_value = False):
		window_num=window_num+1


	n=0
	for key,value in db_feats.RangeIter():
		n=n+1
		if n>1:
			break
		#f_size=len(value)
		datum.ParseFromString(db_feats.Get(key))
		f_size=len(datum.float_data)
		if f_size == 0:
			f_size=len(datum.data)
			is_float_data=False
		print f_size


	n=0
	for key,value in db_label.RangeIter():
		n=n+1
		if n>1:
			break
		#l_size=len(value)
		datum.ParseFromString(value)
		l_size=len(datum.float_data)
	ft = np.zeros((window_num, float(f_size)))
	lb = np.zeros((window_num, float(l_size)))
	#ft = np.zeros((10, float(f_size)))
	#lb = np.zeros((10, float(l_size)))
	count=0
	for key in db_feats.RangeIter(include_value = False):
		datum.ParseFromString(db_feats.Get(key));
		datum_lb.ParseFromString(db_label.Get(key));
		if f_size > 0:
			if is_float_data:
				ft[count, :]=datum.float_data
			else:
				ft[count, :]=datum.data
		lb[count,:]=datum_lb.float_data
		#print ft[count,:]
		print 'convert feature # : %d key is %s' %(count,key)
		count=count+1
		#print 'count total number is: %f' %(count)
		#if count > 10:
		#	break
	#total_subNum = range(10)
	sub_count = count/200
	#print 'sub_count is %f' %(sub_count)
	for i in range(200):
		i
		print 'time 1: %f' %(time.time() - start)
		start_idx = i*sub_count
		if i<199:
			end_idx = start_idx+sub_count
		else:
			end_idx = count
		data = {u'feat_label' : {
			u'feat' : ft[start_idx:end_idx,:],
		 	u'label' : lb[start_idx:end_idx,:],
		 	}
		 }
		# data = {u'feat_label' : {
		# 	u'feat' : ft[i:,:],
		# 	u'label' : lb[i,:],
		# 	}
		# }
		print 'save result to : %s'%(mat_file)
		new_mat_file=mat_file[:-4]+'_'+ str(i)+'.mat'
		hdf5storage.savemat(new_mat_file,data, format='7.3')
		print 'time 2: %f' %(time.time() - start)
		print 'done!'

if __name__ == "__main__":
   main(sys.argv[1:])
