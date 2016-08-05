import os 
import re
import pandas as pd 

def clean_train(TRAIN_DATA_PATH = "path to trainfile", out_train_file_name = "train.csv"):
	print "start cleaning train data"
	train_file_names = os.listdir(TRAIN_DATA_PATH)
	train_data_list = []
	for train_file_name in train_file_names:
		if not train_file_name.endswith(".txt"):
			continue
		train_file = os.path.join(TRAIN_DATA_PATH,train_file_name)

		label = int(train_file_name[0])
		with open(train_file,'r') as f:
			lines = f.read().splitlines()

		labels = [label] * len(lines)

		label_series = pd.Series(labels)
		lines_series = pd.Series(lines)

		#construct dataframe
		data_pd = pd.concat([label_series,lines_series],axis = 1)
		train_data_list.append(data_pd)
	train_data_pd = pd.concat(train_data_list,axis =0)
	
	#output train data
	train_data_pd.columns = ['label','text']
	train_data_pd.to_csv(os.path.join(TRAIN_DATA_PATH,out_train_file_name),index = None, encoding = 'utf-8',header = True)

def clean_test(TEST_DATA_PATH = "path to testfile",test_file_name = "test.csv",out_test_file_name = "test.csv"):
	print 'start cleaning test data'
	test_file = os.path.join(TEST_DATA_PATH,test_file_name)
	with open(test_file,'r') as f:
		lines = f.read(),splitlines()

	lines_series = pd.Series(lines)

	test_data_list = pd.Series(lines_series,neme = 'text')

	#output test data
	test_data_list.to_csv(os.path.join(TEST_DATA_PATH,out_test_file_name), index = None,encoding = 'utf-8',header = True)

