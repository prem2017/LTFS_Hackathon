# -*- coding: utf-8 -*-

""" ©Prem Prakash
	Machine Learning model for training and prediction.
	This module only contains model based on Naive Bayes.
"""


import sys
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer

# Classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)



import sklearn.metrics
import sklearn.pipeline
from data_loader import PreprocessedData


import util



class NaNCategoricalTransformer(BaseEstimator, TransformerMixin):
	"""
	Replace NaN with string 'nan'
	"""
	
	def fit(self, X, y):
		# print('[NaNCategoricalTransformer] fit')
		return self
	
	def transform(self, x):
		# print('[NaNCategoricalTransformer] transform')
		return x.fillna('nan')


#
class NaNContinuousValTransformer(BaseEstimator, TransformerMixin):
	"""
	Replace NaN with zero.
	"""
	
	def fit(self, X, y):
		# print('[NaNContinuousValTransformer] fit')
		return self
	
	def transform(self, x):
		# print('[NaNContinuousValTransformer] transform')
		return x.fillna(0)


#
class MvalueContinuousTransformer(BaseEstimator, TransformerMixin):
	"""
	Replace missing variable with mean of their respective column values
	"""
	
	def fit(self, X, y):
		# print('[MvalueContinuousTransformer] fit')
		return self
	
	def transform(self, x):
		# print('[MvalueContinuousTransformer] transform')
		xcol_mean = x.mean()
		for k, v in xcol_mean.items():
			x[k] = x[k].replace(0, v)
		return x


# https://stackoverflow.com/questions/31259891/put-customized-functions-in-sklearn-pipeline
class CustomNBModel(object):
	"""
	This is a CustomNBModel composed of two or more naive bayes models.
	Two separate pipelines to handle:
		1. Categorical data using MultinomialNB
		2. Continuous data using GaussianNB
		Note: Output from these two NB model are again
			  fed into GaussianNB classifier but might not have best perfomance.
	"""
	
	
	def __init__(self):
		# super(CustomPipeline, self).__init__(steps, memory)

		# Only on categorical data
		self.cat_clf = MultinomialNB() # Pipeline([('cat_data_clf', MultinomialNB())], memory=None)
		
		# Only on continuous data
		self.cont_clf = GaussianNB() # Pipeline([('cont_data_clf', GaussianNB())], memory=None)
		
		# For cobining the output of categorical and continuous
		self.combined_clf = GaussianNB()
		
	def set_cat_clf(self, model):
		self.cat_clf = model
	
	def set_cont_clf(self, model):
		self.cont_clf = model
		
	def set_combined_clf(self, model):
		self.combined_clf = model
		
	def set_all_clf(self, model_cat, model_cont, model_comb):
		self.cat_clf = model_cat
		self.cont_clf = model_cont
		self.combined_clf = model_comb
		
	
	# Override the fit method which internally call its member pipelines
	def fit(self, x, y):
		x_catg, x_cont = x
		
		
		self.cat_clf.fit(x_catg, y)
		util.PickleHandler.dump_in_pickle(self.cat_clf, util.NaiveBayesModelUtil.CATEGORICAL_MODEL_PATH)

		self.cont_clf.fit(x_cont, y)
		util.PickleHandler.dump_in_pickle(self.cont_clf, util.NaiveBayesModelUtil.CONTINUOUS_MODEL_PATH)

		
		predict_proba_catg_part = self.cat_clf.predict_proba(x_catg)
		predict_proba_conti_part = self.cont_clf.predict_proba(x_cont)
		
		# TODO: different ways to combine check impact
		# combined_features = np.hstack((predict_proba_catg_part[:, [1]], predict_proba_conti_part[:, [1]]))
		combined_features = np.hstack((predict_proba_catg_part, predict_proba_conti_part))

		# GaussianNB on accumulated output from the other
		self.combined_clf.fit(combined_features, y)
		util.PickleHandler.dump_in_pickle(self.combined_clf, util.NaiveBayesModelUtil.COMBINED_MODEL_PATH)
		
		return self
	
	# override the predict_proba method
	def predict_proba(self, x):
		x_catg, x_cont = x
		
		
		cat_prob = predict_proba_catg_part = self.cat_clf.predict_proba(x_catg)
		cont_prob = predict_proba_conti_part = self.cont_clf.predict_proba(x_cont)
		
		# TODO: different ways to combine check impact
		# combined_features = np.hstack((predict_proba_catg_part[:, [1]], predict_proba_conti_part[:, [1]]))
		combined_features = np.hstack((predict_proba_catg_part, predict_proba_conti_part))

		# Predict from GaussianNB on combined output
		comb_prob = final_predict_proba = self.combined_clf.predict_proba(combined_features)
		
		
		
		# Multiply the probability from MultinomialNB and GaussianNB # Also and individually as well for catg part of data and conti part of data
		# TODO: For now (0, 0) is 0 and (1,1) is 1 but it could be or operation
		# cat_cont_mult = predict_proba_catg_part[:, 1] * predict_proba_conti_part[:, 1]
		cat_cont_mult_plain  = cat_prob * cont_prob
		normalizer_col =  cat_cont_mult_plain.sum(axis=1)
		normalizer = np.hstack((normalizer_col.reshape(-1, 1), normalizer_col.reshape(-1, 1)))
		cat_cont_mult = (cat_cont_mult_plain / normalizer)
		
		
		proba_dict = {'cat': cat_prob[:, 1],
		              'cont': cont_prob[:, 1],
		              'comb': comb_prob[:, 1],
		              'cat_cont_mult': cat_cont_mult[:, 1]}
		
		return proba_dict # predict_proba_conti_part # predict_proba_catg_part # final_predict_proba_mult # final_predict_proba

	
	@staticmethod
	def get_model_class_from_pickle():
		model_cat = util.PickleHandler.extract_from_pickle(util.NaiveBayesModelUtil.CATEGORICAL_MODEL_PATH)
		model_cont= util.PickleHandler.extract_from_pickle(util.NaiveBayesModelUtil.CONTINUOUS_MODEL_PATH)
		model_comb = util.PickleHandler.extract_from_pickle(util.NaiveBayesModelUtil.COMBINED_MODEL_PATH)
		
		model_obj = CustomNBModel()
		model_obj.set_all_clf(model_cat, model_cont, model_comb)
		return model_comb




class Trainer(object):

	
	@staticmethod
	def check_nb_model(lenght=50000):
		cat_data, cont_data = PreprocessedData.get_split_data(train=True)
		
		X_cat, Y, uid = cat_data
		X_cont, _, _ = cont_data # uncaptured 2nd and 3rd are same as above
		
		num_rows = X_cat.shape[0]
		indexes = np.arange(num_rows)
		np.random.shuffle(indexes)
		
		test_len = 10000
		X_cat_train, X_cat_test = X_cat[indexes[0:lenght], :], X_cat[indexes[lenght:lenght+test_len], :]
		X_cont_train, X_cont_test = X_cont[indexes[0:lenght], :], X_cont[indexes[lenght:lenght+test_len], :]
		Y_train, Y_test = Y.values[indexes[0:lenght]], Y.values[indexes[lenght:lenght+test_len]]

	
		custom_model = CustomNBModel()
		custom_model.fit((X_cat_train, X_cont_train), Y_train)
		
		ytrain_prob_dict = custom_model.predict_proba((X_cat_train, X_cont_train))
		ytest_prob_dict = custom_model.predict_proba((X_cat_test, X_cont_test))
		
		
		def compute_metrics(y_true, yprob_dict, type='Train'):
			print('\n\n\n\n[{}] => '.format(type))
			for clf, y_pred in yprob_dict.items():
				print('\n\n[{}] ROC = {}, F1-Score = {},\n Classification report = \n{}'.format(clf,
				                                              util.MetricBinaryClass.roc_metric(y_true, y_pred),
				                                              util.MetricBinaryClass.f1_value(y_true, y_pred),
				                                              util.MetricBinaryClass.classification_report(y_true, y_pred)))
		
		compute_metrics(Y_train, ytrain_prob_dict, 'Train')
		compute_metrics(Y_test, ytest_prob_dict, type='Test')
			
		
	@staticmethod
	def train_models():
		cat_data, cont_data = PreprocessedData.get_split_data(train=True)
		X_cat, Y, uid = cat_data
		X_cont, _, _ = cont_data # uncaptured 2nd and 3rd are same as above
		
		custom_model = CustomNBModel()
		custom_model.fit((X_cat, X_cont), Y)
		
		return custom_model
	
	@staticmethod
	def predict_on_test(model):
		cat_data, cont_data = PreprocessedData.get_split_data(train=False)
		X_cat, Y, uid = cat_data
		X_cont, _, uid_cont = cont_data
		
		ytest_prob_dict = model.predict_proba((X_cat, X_cont))
		
		result_dir_path = util.get_result_dir()
		sample_result_file = util.get_sample_submission_file()
		df = pd.read_csv(filepath_or_buffer=sample_result_file)
		
		submission_uid = df['UniqueID']
		# assert (uid.values == submission_uid.values).sum() == X_cat.shape[0], 'Id should match exactly'
		
		for key, ypred_prob  in ytest_prob_dict.items():
			print('Saving for {}'.format(key))
			df['loan_default'] = np.round(ypred_prob, 6)
			save_path = util.get_result_dir() + key + '_pred.csv'
			df.to_csv(path_or_buf=save_path, sep=',', index=None)
		


def sample_submission_gen(fname='sample_submission.csv'):
	sample_fpath = util.get_result_dir() + fname
	submmision_path = util.get_result_dir() + 'check_submission.csv'
	df = pd.read_csv(filepath_or_buffer=sample_fpath)
	
	# for i in range(len(df.iloc[:, 1])):
	# 	df[i, 1] = np.round(np.random.uniform(0, 1), 4)
	len_val = df.shape[0]
	probs = np.round(np.random.uniform(0, 1, len_val), 4)
	df['loan_default'] = probs
	df.to_csv(path_or_buf=submmision_path, sep=',', index=None)
	print('Hello')




if __name__ == '__main__':
	print('[Using SKLearn]')
	# print(score_solution())
	# Trainer.check_nb_model()
	# sample_submission_gen()
	model = Trainer.train_models()
	Trainer.predict_on_test(model)
	print('********************************* Complete *********************************')