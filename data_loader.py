# -*- coding: utf-8 -*-

""" ©Prem Prakash
	Module for loading and preprocessing the dataset. 
"""


import os
import sys
from copy import deepcopy

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin




import numpy as np
import h5py

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import util





class NoTransformerException(Exception):
	"""docstring for NoTransformerException
	"""
	pass

# ---------------------------- Transformers
class NaNCategoricalTransformer(object):
	"""
	Replace NaN with string 'nan'
	"""
	
	@staticmethod
	def transform(x: pd.DataFrame):
		# print('[NaNCategoricalTransformer] transform')
		return x.fillna('nan')


#
class NaNContinuousValTransformer(object):
	"""
	Replace NaN with zero. and then mean value if required
	"""
	
	@staticmethod
	def transform(x: pd.DataFrame):
		# print('[NaNContinuousValTransformer] transform')
		# x = x.fillna(None)
		xcol_mean = x.mean()
		for k, v in xcol_mean.items():
			x[k].fillna(v, inplace=True)
		return x


	
class CategoryColumnTransformer(object):
	"""docstring for CategoryColumnTransformer : helper: https://stackoverflow.com/questions/31259891/put-customized-functions-in-sklearn-pipeline
		# TODO: docstring
	"""
	transformer_object_path = util.get_base_datadir() + 'column_transformer.pkl'
	
	@classmethod
	def category_transformer_exists(cls):
		return os.path.exists(cls.transformer_object_path)
	
	def __init__(self, category_columns=None):
		super(CategoryColumnTransformer, self).__init__()
		self.category_columns = category_columns
		self.category_column_transformer = None
		
		if self.category_columns:
			# This helps in creating columns for each of the categorical columns with column-name => column_name + '_category' + '{val}' for example salary_category_high where vals are high, low, medium
			transformers_list = [(col_name + '_category', CountVectorizer(analyzer=lambda x: [str(x)]), col_name) for col_name in category_columns]
		else:
			# TODO: Category cloumns
			transformers_list = None
		
		if transformers_list:
			self.category_column_transformer = ColumnTransformer(transformers=transformers_list, remainder='passthrough')
		else:
			print('transformers=[] is empty')
			self.category_column_transformer = ColumnTransformer(transformers=[], remainder='passthrough')
		
	
	def fit(self, X):
		self.fit_transform(X)
		return self
		
	def fit_transform(self, X, dense=False):
		print('[Before Fit Transform]', X.shape)
		assert self.category_column_transformer is not None, '[Assertion Error] Column Transformer is None'
		print('Fitting transform ...')
		x_t = self.category_column_transformer.fit_transform(X)
		
		# util.PickleHandler.dump_in_pickle(self.category_column_transformer, CategoryColumnTransformer.transformer_object_path)
		print('[After Fit Transform]', x_t.shape)

		return x_t.todense() if dense else x_t
		
	def transform(self, X, dense=False):
		print('[Before Fit Transform]', X.shape)
		x_t = self.category_column_transformer.transform(X)
		print('[After Transform]', x_t.shape)
		return x_t.todense() if dense else x_t
	
	@classmethod
	def load_transformer_from_mem(cls):
		exists = cls.category_transformer_exists()
		if exists:
			return util.PickleHandler.extract_from_pickle(cls.transformer_object_path)
		else:
			raise NoTransformerException('No column transformer is saved on hard disk.')
	
	
	@classmethod
	def transform_from_fitted_transformer(cls, X, dense=False):
		transformer = cls.load_transformer_from_mem()
		return transformer.transform(X).todense() if dense else transformer.transform(X)
		
	
		
class DataPreprocessor(object):
	
	@staticmethod
	def get_data(has_label = True):
		names = ['UniqueID', 'disbursed_amount', 'asset_cost', 'ltv', 'branch_id', 'supplier_id',
					'manufacturer_id', 'Current_pincode_ID', 'Date.of.Birth', 'Employment.Type',
					'DisbursalDate', 'State_ID', 'Employee_code_ID', 'MobileNo_Avl_Flag',
					'Aadhar_flag', 'PAN_flag', 'VoterID_flag', 'Driving_flag', 'Passport_flag',
					'PERFORM_CNS.SCORE', 'PERFORM_CNS.SCORE.DESCRIPTION', 'PRI.NO.OF.ACCTS',
					'PRI.ACTIVE.ACCTS', 'PRI.OVERDUE.ACCTS', 'PRI.CURRENT.BALANCE', 'PRI.SANCTIONED.AMOUNT',
					'PRI.DISBURSED.AMOUNT', 'SEC.NO.OF.ACCTS', 'SEC.ACTIVE.ACCTS', 'SEC.OVERDUE.ACCTS',
					'SEC.CURRENT.BALANCE', 'SEC.SANCTIONED.AMOUNT', 'SEC.DISBURSED.AMOUNT', 'PRIMARY.INSTAL.AMT',
					'SEC.INSTAL.AMT', 'NEW.ACCTS.IN.LAST.SIX.MONTHS', 'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS',
					'AVERAGE.ACCT.AGE', 'CREDIT.HISTORY.LENGTH', 'NO.OF_INQUIRIES'] + (['loan_default'] if has_label else [])
		
		
		fullpath = util.LTFSDataDesc.train_datapath() if has_label else util.LTFSDataDesc.test_datapath()
		df = pd.read_csv(filepath_or_buffer=fullpath,
						sep=',',
						na_values='?',
						# names=names)
						converters = util.LTFSDataDesc.converter_dict())
		

		# if has_label:
		# 	indexes = np.array(list(range(df.shape[0])))
		# 	pos_cls = (df[util.LTFSDataDesc.LABEL_COL] == 1).values
		# 	pids = indexes[pos_cls]
		# 	pos_cls_len = len(pids)
			
		# 	nids = np.array(list(set(indexes) - set(pids)))
		# 	np.random.shuffle(nids)
		# 	neg_cls_limited_len = min(2*pos_cls_len, df.shape[0] - pos_cls_len)
		# 	nids = nids[:neg_cls_limited_len]

		# 	# Union of both ids
		# 	fids = np.union1d(pids, nids)
		# 	np.random.shuffle(fids)
		# 	df = df.iloc[fids]
		# 	print('[Final] shape = ', df.shape)





		more_3 = ['Aadhar_flag', 'PAN_flag',  'VoterID_flag',  'Driving_flag',  'Passport_flag', 'PERFORM_CNS.SCORE']
		for fname in more_3:
			print('\n\nFeature Name = {}, uniques = {}'.format(fname, df[fname].unique()))
		

		desc = df.describe()
		i = 0
		while i+5 < desc.shape[1]:
			print(desc.iloc[:, i:i+5], '\n\n')
			i += 5
		print(desc.iloc[:, i:])

		# df.columns = df.columns.str.lower()
		# Split into feature matrix X and labels y.
		uid_col_name = util.LTFSDataDesc.UNIQUE_ID
		uid = df[uid_col_name]
		
		
		label_col_name = util.LTFSDataDesc.LABEL_COL
		if has_label:
			print('[Final] shape before extracting y is = ', df.shape)
			y = df[label_col_name]
			print('\n[Positive] = {}, [Negative] = {}'.format(sum(y), len(y) - sum(y)))
			df = df.drop([uid_col_name, label_col_name], axis=1)
		else:
			y = 0
			df = df.drop([uid_col_name], axis=1)
		x = df
		fullpath_converted_df_path = fullpath + '_converted.pkl'
		# [Uncomment]
		util.PickleHandler.dump_in_pickle((x, y, uid, fullpath), fullpath_converted_df_path)
		
		return x, y, uid, fullpath

	
	@staticmethod
	def preprocess_data(X: pd.DataFrame, Y=None, uid=None, fullpath=None, cat_transformer_cat = None, cat_transformer_combined = None):
		
		cat_cols_vectorizable = cat_cols_all = cat_cols_dict = util.LTFSDataDesc.CATEGORICAL_COLUMNS_ALL
		cont_cols_all = util.LTFSDataDesc.CONTINUOUS_COLUMNS_ALL
		
		if isinstance(cat_cols_dict, dict):
			cat_cols_vectorizable = cat_cols_all['vectorizable']
			cat_cols_all = []
			for k, v in cat_cols_dict.items():
				cat_cols_all += v
		
		X_cat_cols = X[cat_cols_all]
		X_cont_cols = X[cont_cols_all]
		
		# NaN handling
		X_cat_cols = NaNCategoricalTransformer.transform(X_cat_cols)
		X_cont_cols = NaNContinuousValTransformer.transform(X_cont_cols)
		
		# Store separate and not dense
		if cat_transformer_cat is None:
			cat_transformer_cat = CategoryColumnTransformer(category_columns=cat_cols_vectorizable)
			X_transformed_categorical = cat_transformer_cat.fit_transform(X_cat_cols, dense=False)
			
			X_transformed_continuous = X_cont_cols.values
		else:
			X_transformed_categorical = cat_transformer_cat.transform(X_cat_cols, dense=False)
			
			X_transformed_continuous = X_cont_cols.values

		cat_fullpath = fullpath + util.LTFSDataDesc.categorical_transformed_extension()
		cont_fullpath = fullpath + util.LTFSDataDesc.continuous_transformed_extension()
		
		# [Save]
		util.PickleHandler.dump_in_pickle((X_transformed_categorical, Y, uid), cat_fullpath)
		util.PickleHandler.dump_in_pickle((X_transformed_continuous, Y, uid), cont_fullpath)
		
		

		# Store combined and dense
		
		X_transformed_combined = pd.concat([X_cat_cols, X_cont_cols], axis=1)
		
		if cat_transformer_combined is None:
			cat_transformer_combined = CategoryColumnTransformer(category_columns=cat_cols_vectorizable)
			X_transformed_combined = cat_transformer_combined.fit_transform(X_transformed_combined, dense=False)
		else:
			X_transformed_combined = cat_transformer_combined.transform(X_transformed_combined, dense=False)
		
		transforemd_combined_fullpath = fullpath + util.LTFSDataDesc.combined_transformed_extension()
		
		# [Save]
		util.PickleHandler.dump_in_pickle((X_transformed_combined, Y, uid), transforemd_combined_fullpath)
		
		
		return cat_transformer_cat, cat_transformer_combined
		

class PreprocessedData(object):
	""" docstring of ProcessedData
	
	"""
	
	@staticmethod
	def get_split_data(train=True):
		if train:
			datapath_cat = util.LTFSDataDesc.train_datapath() + util.LTFSDataDesc.categorical_transformed_extension()
			datapath_cont = util.LTFSDataDesc.train_datapath() + util.LTFSDataDesc.continuous_transformed_extension()
		else:
			datapath_cat = util.LTFSDataDesc.test_datapath() + util.LTFSDataDesc.categorical_transformed_extension()
			datapath_cont = util.LTFSDataDesc.test_datapath() + util.LTFSDataDesc.continuous_transformed_extension()
		
		print('[Cat] = {} \n[Cont] = {}'.format(datapath_cat, datapath_cont))
		
		
		data_cat = util.PickleHandler.extract_from_pickle(datapath_cat)
		data_cont = util.PickleHandler.extract_from_pickle(datapath_cont)
		
		return data_cat, data_cont
	
	@staticmethod
	def get_combined_data(train=True):
		if train:
			datapath_combined = util.LTFSDataDesc.train_datapath() + util.LTFSDataDesc.combined_transformed_extension()
		else:
			datapath_combined = util.LTFSDataDesc.test_datapath() + util.LTFSDataDesc.combined_transformed_extension()
		print('[Combined] = {}'.format(datapath_combined))
		
		data_combined = util.PickleHandler.extract_from_pickle(datapath_combined)
		
		return datapath_combined

	
	

if __name__ == '__main__':
	print('[DataLoader]')
	train_data = DataPreprocessor.get_data(has_label=True)
	test_data = DataPreprocessor.get_data(has_label=False)
	# print('Preliminary conversion done')
	# # when already preliminary processing done
	# train_data = util.PickleHandler.extract_from_pickle(util.get_base_datadir() + '/train.csv_converted.pkl')
	# test_data = util.PickleHandler.extract_from_pickle(util.get_base_datadir() + '/test.csv_converted.pkl')
	
	cat_transformer_cat, cat_transformer_combined = DataPreprocessor.preprocess_data(*train_data)
	cat_transformer_cat, cat_transformer_combined = DataPreprocessor.preprocess_data(*test_data, cat_transformer_cat=cat_transformer_cat, cat_transformer_combined=cat_transformer_combined)
	
	util.PickleHandler.dump_in_pickle(cat_transformer_cat, 'cat_transformer_cat.pkl')
	util.PickleHandler.dump_in_pickle(cat_transformer_combined, 'cat_transformer_combined.pkl')

	
	print('*********************** Preprocessing Completed ***********************')