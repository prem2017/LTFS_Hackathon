# -*- coding: utf-8 -*-


""" ©Prem Prakash
	Utility module used throughout the project
"""


import os
from datetime import datetime, date
import re
import pickle
import sys

import numpy as np
import sklearn


PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_DATAPATH = os.path.join(PROJECT_PATH, 'data/')
get_base_datadir = get_base_datapath = lambda : BASE_DATAPATH


RESULT_DIR = os.path.join(PROJECT_PATH, 'result_dir/')
get_result_dir = lambda : RESULT_DIR
get_sample_submission_file = lambda : get_result_dir() + 'sample_submission.csv'


MODEL_ROOT_DIR = os.path.join(PROJECT_PATH, 'trained_model/')

class ColumnConverter(object):
	"""docstring for ColumnConverter
		Different converter for preprocessing data
	"""
	
	@staticmethod
	def calculate_year(dob):
		return ColumnConverter.calculate_time_length(dob, 'year')
	
	@staticmethod
	def calculate_months(tstamp):
		return ColumnConverter.calculate_time_length(tstamp, 'month')
	
	@staticmethod
	def calculate_time_length(time_form, type: str):
		short_strf = '%d-%m-%y'
		long_strf = '%d-%m-%Y'
		date_format_dict = {8: short_strf , 10: long_strf, }
		
		
		if isinstance(time_form, str):
			# print('\nDict = {} Tstamp = {}, type(tstamp) = {}'.format(date_format_dict, time_form, isinstance(time_form, str)))
			time_form = datetime.strptime(time_form, date_format_dict[len(time_form)])
		today = date.today()
		
		if type == 'year':
			return abs(today.year - time_form.year) - ((today.month, today.day) < (time_form.month, time_form.day))
		else: # type == 'month'
			return (12 * abs(today.year - time_form.year)) - (time_form.month - today.month)
	
	@staticmethod
	def calculate_months_from_str(yrs_mon):
		yrs_mon = re.findall(r'\d+', yrs_mon)
		multipliers = {0: 12, 1: 1}
		months = 0
		for i, val in enumerate(yrs_mon):
			months += (int(val)  * multipliers[i])
		
		return months
			




# ----- Save and Extract from Pickle ------
class PickleHandler(object):
	
	@staticmethod
	def dump_in_pickle(py_obj, filepath):
		with open(filepath, 'wb') as pfile:
			pickle.dump(py_obj, pfile)
	
	@staticmethod
	def extract_from_pickle(filepath):
		with open(filepath, 'rb') as pfile:
			py_obj = pickle.load(pfile)
			return py_obj


class LTFSDataDesc(object):
	train_datapath = lambda : os.path.join(BASE_DATAPATH, 'train.csv')
	test_datapath = lambda : os.path.join(BASE_DATAPATH, 'test.csv')
	
	categorical_transformed_extension = lambda : '_transformed_categorical.pkl'
	continuous_transformed_extension = lambda : '_transformed_continuous.pkl'
	
	combined_transformed_extension = lambda : '_transformed_combined.pkl'
	
	
	UNIQUE_ID = 'UniqueID'
	LABEL_COL = 'loan_default'
	CATEGORICAL_COLUMNS_ALL = {	'vectorizable': ['branch_id', 'supplier_id', 'manufacturer_id', 'Current_pincode_ID',
											'Employment.Type','State_ID', 'Employee_code_ID', 'PERFORM_CNS.SCORE.DESCRIPTION'],
							'vectorized': ['MobileNo_Avl_Flag', 'Aadhar_flag', 'PAN_flag', 'VoterID_flag', 'Driving_flag','Passport_flag']
							}
	
	
	CONTINUOUS_COLUMNS = ['disbursed_amount', 'asset_cost', 'ltv', 'PERFORM_CNS.SCORE',
							'PRI.NO.OF.ACCTS', 'PRI.ACTIVE.ACCTS', 'PRI.OVERDUE.ACCTS', 'PRI.CURRENT.BALANCE',
							# May need to take difference of them
							'PRI.SANCTIONED.AMOUNT', 'PRI.DISBURSED.AMOUNT','PRIMARY.INSTAL.AMT',
							'SEC.NO.OF.ACCTS', 'SEC.ACTIVE.ACCTS', 'SEC.OVERDUE.ACCTS', 'SEC.CURRENT.BALANCE',
							'SEC.SANCTIONED.AMOUNT', 'SEC.DISBURSED.AMOUNT', 'SEC.INSTAL.AMT',
							'NEW.ACCTS.IN.LAST.SIX.MONTHS', 'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS', 'NO.OF_INQUIRIES'
						  ]
	
	
	TIMESTAMP_COLUMNS_YEAR = ['Date.of.Birth']
	TIMESTAMP_COLUMNS_MONTH = ['DisbursalDate']
	
	TIME_REGEX_COLUMNS = ['AVERAGE.ACCT.AGE', 'CREDIT.HISTORY.LENGTH']
	
	CONTINUOUS_COLUMNS_ALL = CONTINUOUS_COLUMNS + TIMESTAMP_COLUMNS_YEAR + TIMESTAMP_COLUMNS_MONTH + TIME_REGEX_COLUMNS
	
	
	@staticmethod
	def converter_dict():
		converter = {}
		for k in LTFSDataDesc.TIMESTAMP_COLUMNS_YEAR:
			converter[k] = ColumnConverter.calculate_year
		
		for k in LTFSDataDesc.TIMESTAMP_COLUMNS_MONTH:
			converter[k] = ColumnConverter.calculate_months
		for k in LTFSDataDesc.TIME_REGEX_COLUMNS:
			converter[k] =  ColumnConverter.calculate_months_from_str
	
		return converter


class NaiveBayesModelUtil(object):
	""" docstring for NBModelUtil
	
	"""
	ROOT_NB_MODEL_DIR = os.path.join(MODEL_ROOT_DIR, 'naive_bayes/')
	
	CATEGORICAL_MODEL_PATH =  os.path.join(ROOT_NB_MODEL_DIR, 'categorical_model.pkl')
	CONTINUOUS_MODEL_PATH =  os.path.join(ROOT_NB_MODEL_DIR, 'continuous_model.pkl')
	
	
	COMBINED_MODEL_PATH = os.path.join(ROOT_NB_MODEL_DIR, 'combined_model.pkl')


class MetricBinaryClass(object):
	""" docstring for computing different way to evaluate the model
	
	"""
	
	@staticmethod
	def check_assertion(y_true: np.ndarray, y_pred: np.ndarray):
		assert (y_true.ndim == 1) or (y_true.ndim == 2 and y_true.shape[1] == 1), 'The true label must have only column'
		assert (y_pred.ndim == 1) or (y_pred.ndim == 2 and y_pred.shape[
			1] == 2), 'Predicted probabilities dimension must match sklearn shape'
		y_true = y_true if y_true.ndim == 1 else y_true[:, 0]
		y_pred = y_pred if y_pred.ndim == 1 else y_pred[:, 1]
		return y_true, y_pred
	
		
	@staticmethod
	def roc_metric(y_true: np.ndarray, y_pred: np.ndarray, get_optimal_threshold=False):
		y_true, y_pred = MetricBinaryClass.check_assertion(y_true, y_pred)
		
		roc_score = sklearn.metrics.roc_auc_score(y_true, y_pred)
		if get_optimal_threshold:
			fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_pred)
			tpr_minus_fpr = tpr - fpr
			optimal_idx = np.argmax(tpr_minus_fpr)
			optimal_fpr, optimal_tpr = fpr[optimal_idx], tpr[optimal_idx]
			optimal_threshold = thresholds[optimal_idx]
			return roc_score, optimal_threshold
		else:
			return roc_score
	
	@staticmethod
	def f1_value(y_true: np.ndarray, y_pred: np.ndarray, threshold=0.5):
		y_true, y_pred = MetricBinaryClass.check_assertion(y_true, y_pred)
		y_pred = (y_pred >= threshold) + 0
		return sklearn.metrics.f1_score(y_true, y_pred)
	
	@staticmethod
	def classification_report(y_true: np.ndarray, y_pred: np.ndarray, threshold=0.5):
		y_true, y_pred = MetricBinaryClass.check_assertion(y_true, y_pred)
		y_pred = (y_pred >= threshold) + 0
		return sklearn.metrics.classification_report(y_true, y_pred)
	
	@staticmethod
	def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, threshold=0.5):
		y_true, y_pred = MetricBinaryClass.check_assertion(y_true, y_pred)
		y_pred = (y_pred >= threshold) + 0
		return sklearn.metrics.confusion_matrix(y_true, y_pred)
	
	@staticmethod
	def pr_score(y_true: np.ndarray, y_pred: np.ndarray):
		y_true, y_pred = MetricBinaryClass.check_assertion(y_true, y_pred)
		# Similar to ROC we can find optimal threshold for this also but is not necessary
		return sklearn.metrics.precision_score(y_true, y_pred)
	

	

	
	
	
URL = 'https://datahack.analyticsvidhya.com/contest/ltfs-datascience-finhack-an-online-hackathon/'

if __name__ == '__main__':
	print('Util Module')
	print(ColumnConverter.calculate_time_length('01-06-68', type='year'))
	print(LTFSDataDesc.CONTINUOUS_COLUMNS_ALL)