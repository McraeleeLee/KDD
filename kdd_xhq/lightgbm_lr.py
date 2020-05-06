# coding: utf-8
# pylint: disable = invalid-name, C0111
from __future__ import division
import json
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

from util.util import *

# specify your configurations as a dict
LGB_PARAMETERS = {
	'seed': 120,
	'feature_fraction_seed': 120,
	'bagging_seed': 120,
	'drop_seed': 120,
	'data_random_seed': 120,
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 63,
   	'num_trees': 100,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
   	'device': 'gpu',
	'gpu_platform_id': 0,
	'gpu_device_id': 0,
}

# number of leaves,will be used in feature transformation
NUM_LEAVES = 63


def gbdt_lr(stage, split):
	filename = "dataset/construct_split{}_click-{}_%s.pkl".format(split, stage)
	# filename = "dataset/example_click-{}_%s.pkl".format(stage)
	
	# load or create your dataset
	print('Load data...')
	df_train = pickle_read(filename % "train")
	df_valid = pickle_read(filename % "valid")
	df_train = np.array(df_train)
	df_valid = np.array(df_valid)

	# df_train = pd.read_csv(filename % "train", header=None, sep=' ')
	# df_valid = pd.read_csv(filename % "valid", header=None, sep=' ')

	""" TODO: decide the input and output for KDD """
	# X: times tamp, user age level, user gender(one-hot), user city level, item txt, item img (261 dim)
	# Y: 1 (regression value)
	# file format: Y, ...X
	y_train = df_train[:, 0]  # training label
	y_valid = df_valid[:, 0]   # validing label
	X_train = df_train[:, 1:]  # training dataset
	X_valid = df_valid[:, 1:]  # validing dataset

	# y_train = df_train[0]  # training label
	# y_valid = df_valid[0]   # validing label
	# X_train = df_train.drop(0, axis=1)  # training dataset
	# X_valid = df_valid.drop(0, axis=1)  # validing dataset

	# create dataset for lightgbm
	lgb_train = lgb.Dataset(X_train, y_train)
	# lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

	print('Start training...')
	# train
	gbm = lgb.train(LGB_PARAMETERS,
                 lgb_train,
                 num_boost_round=100,
                 valid_sets=lgb_train)

	print('Save model...')
	# save model to file
	model_folder = "lightgbm_model"
	try:
		os.mkdir(model_folder)
	except:
		pass
	gbm.save_model(os.path.join(model_folder, 'model.txt'))

	print('Start predicting...')
	# predict and get data on leaves, training data
	y_pred = gbm.predict(X_train, pred_leaf=True)
	# print(type(y_pred), y_pred.shape)

	# feature transformation and write result
	print('Writing transformed training data')
	transformed_training_matrix = np.zeros(
		[len(y_pred), len(y_pred[0]) * NUM_LEAVES], dtype=np.int64)
	for i in range(0, len(y_pred)):
		temp = np.arange(len(y_pred[0])) * NUM_LEAVES - 1 + np.array(y_pred[i])
		transformed_training_matrix[i][temp] += 1

	#for i in range(0,len(y_pred)):
	#	for j in range(0,len(y_pred[i])):
	#		transformed_training_matrix[i][j * NUM_LEAVES + y_pred[i][j]-1] = 1

	# predict and get data on leaves, validing data
	y_pred = gbm.predict(X_valid, pred_leaf=True)

	# feature transformation and write result
	print('Writing transformed validation data')
	transformed_validing_matrix = np.zeros(
		[len(y_pred), len(y_pred[0]) * NUM_LEAVES], dtype=np.int64)
	for i in range(0, len(y_pred)):
		temp = np.arange(len(y_pred[0])) * NUM_LEAVES - 1 + np.array(y_pred[i])
		transformed_validing_matrix[i][temp] += 1

	#for i in range(0,len(y_pred)):
	#	for j in range(0,len(y_pred[i])):
	#		transformed_validing_matrix[i][j * NUM_LEAVES + y_pred[i][j]-1] = 1

	print('Calculate feature importances...')
	# feature importances
	print('Feature importances:', list(gbm.feature_importance()))
	print('Feature importances:', list(gbm.feature_importance("gain")))

	# Logestic Regression Start
	print("Logestic Regression Start")

	# load or create your dataset
	print('Load data...')
	# transformed_training_matrix = preprocessing.scale(transformed_training_matrix)
	# transformed_validing_matrix = preprocessing.scale(transformed_validing_matrix)

	c = np.array([1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001])
	for _c in c:
		lm = LogisticRegression(penalty='l2', C=_c, max_iter=500)  # logestic model construction
		lm.fit(transformed_training_matrix, y_train)  # fitting the data

		#y_pred_label = lm.predict(transformed_training_matrix )  # For training data
		#y_pred_label = lm.predict(transformed_validing_matrix)    # For validing data
		#y_pred_est = lm.predict_proba(transformed_training_matrix)   # Give the probabilty on each label
		# Give the probabilty on each label
		y_pred_est = lm.predict_proba(transformed_validing_matrix)

	#print('number of validing data is ' + str(len(y_pred_label)))
	#print(y_pred_est)

	# calculate predict accuracy
		#num = 0
		#for i in range(0,len(y_pred_label)):
		#if y_valid[i] == y_pred_label[i]:
		#	if y_train[i] == y_pred_label[i]:
		#		num += 1
		#print('penalty parameter is '+ str(_c))
		#print("prediction accuracy is " + str((num)/len(y_pred_label)))

		# Calculate the Normalized Cross-Entropy
		# for validing data
		# NE = (-1) / len(y_pred_est) * sum(((1+y_valid)/2 *
        #                              np.log(y_pred_est[:, 1]) + (1-y_valid)/2 * np.log(1 - y_pred_est[:, 1])))
		# for training data
		#NE = (-1) / len(y_pred_est) * sum(((1+y_train)/2 * np.log(y_pred_est[:,1]) +  (1-y_train)/2 * np.log(1 - y_pred_est[:,1])))
		print("c {}: Normalized Cross Entropy".format(_c), log_loss(y_valid, y_pred_est, labels=[.0, 1.0]))


if __name__ == "__main__":
	gbdt_lr(stage=0, split=1)
