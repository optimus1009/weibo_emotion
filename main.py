#! usr/bin/python

import jieba
import pandas as pd
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from cleandata import clean_train, clean_test
from sklearn.ensemble.forest import RandomForestClassfier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def words_to_features(raw_line,stopwords_path = 'path to stop words'):
	stopwords = {}.fromkeys([line.rstrip() for line in open(stopwords_path)])
	chinese_only = raw_line
	word_lst = jieba.cut(chinese_only)
	meaninful_words = []
	for word in word_lst:
		word = word.encode('utf8')
		if word not in stopwords:
			meaninful_words.append(word)
	return ' '.join(meaninful_words)

def drawfeature(TRAIN_DATA_PATH = ' path to train file ', train_file_name = 'train.csv',TEST_DATA_PATH = 'path to test file',test_file_name = 'test.csv'):
	
	train_file = os.path.join(TRAIN_DATA_PATH,train_file_name)
	train_data = pd.read_csv(train_file)
	n_data_train = test_data['text'].size

	test_file = os.path.join(TEST_DATA_PATH,test_file_name)
	test_data = pd.read_csv(test_file)
	n_data_test = test_data['test'].size

	#bag of words model and tf-idf
	vectorizer = CountVectorizer(analyzer = 'word',tokenizer = None, preprocessor = None, stop_words = None,max_feature = 5000)
	transformer = TfidfTransformer()

	#train
	print "start cut words in train data set"
	train_data_words = []
	for i in xrange(n_data_train):
		if((i + 1)%1000 == 0):
			print "drawfeatures line %d of %d " % (i+1,n_data_train)
		train_data_words.append(words_to_features(train_data['text'][i]))

	#draw lables
	#train_data_label = pd.Series(train_data['label'],name = 'label')
	#train_data_labels.to_csv(os.path.join(TRAIN_DATA_PATH,"train_data_label.csv"), index =None,header = True)
	
	print "start bag of words in train dataset"
	#draw features
	train_data_features = vectorizer.fit_transform(train_data_words)
	train_data_features = train_data_features.toarray()
	#train_data_features = pd.DataFrame(train_data_features)
	#train_data_features.to_csv(os.path.join(TRAIN_DATA_PATH,"train_data_features1.csv"), index = None, header = None,encoding = 'utf-8')
	print "start tfidf in train dataset"
	train_data_features = transformer.fit_transform(train_data_features)
	train_data_features = train_data_feature.toarray()
	#train_data_features = pd.DataFrame(train_data_features)
	#train_data_features.to_csv(od.path.join(TRAIN_DATA_PATH,"train_data_features2.csv"),index = None,header = None,encoding = "utf-8")

	#test
	print "start cut words in test dataset"
	test_data_word = []
	for in in xrange(n_data_test):
		if((i + 1)%1000 == 0):
			print "drawfeatures line %d of %d" % (i+1,n_data_test)
		test_data_words.append(words_to_features(test_data['test'][i]))
	#draw features
	print "start bag of words in test dataset"
	test_data_features = vectorizer.fit_transform(test_data_words)
	test_data_features = test_data_features.toarray()
	#test_data_features = pd.DataFrame(test_data_features)
	#test_data_features.to_csv(os.path.join(TEST_DATA_PATH,"test_data_features1.csv"),index = None,header = None,encoding = 'utf-8')
	print "start tfidf in test dataset"
	test_data_features = transformer.fit_transform(test_data_features)
	test_data_features = test_data_features.toarray()
	#test_data_features = pd.DataFrame(test_data_features)
	#test_data_features.to_csv(os.path.join(TEST_DATA_PATH,"test_data_features2.csv"),index = None,header = None,ecoding = "utf-8")

	#random forest
	print "random forest"
	forest = RandomForestClassfier(n_estimators = 100)
	forest = forest.fit(train_data_features,train_data['lebel'])
	pred = forest.predict(test_data_features)
	pred = pd.Series(pred,name = "TARGET")
	pred.to_csv("BOW_TFIDF_RF.csv",index = None,header = True)

	#multinomial naive bayes 
	print "multinomal bayes"
	mnb = MultinomialNB(alpha = 0.01)
	mnb = mnb.fit(train_data_features,train_data["label"])
	pred = mnb.predict(test_data_features)
	pred = pd.Series(pred,name = "TARGET")
	pred.to_csv("BOW_TFIDFRF_MNB.csv",index = None,header = True)

	#KNN
	pirnt "knn"
	knn = KNeighborsClassifier()
	knn = knn.fit(train_data_features,train_data["label"])
	pred = knn.predict(test_data_features)
	pred.to_csv("KNN_TFIDF_KNN.csv",index = None,header = True)

	#svm
	print "SVM"
	svm = SVC(kernel = "linear")
	svm = svm.fit(train_data_features,train_data["lable"])
	pred = svm.predict(test_data_features)
	pred = pd.Series(pred,name = "TARGET")
	pred.to_csv("BOW_TFIDFRF_SVM.csv",index = None,header = True)

drawfeature()