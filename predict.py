import numpy as np
from tqdm import tqdm
from xgboost.sklearn import XGBClassifier, XGBRegressor
import lightgbm as lgb
import argparse, pickle
np.random.seed(-class_idx)


def objToFile(obj,path):
	print "Writing to {}".format(path)
	with open(path,'wb') as f:
			pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def objFromFile(path):
	with open(path,'rb') as f:
		return pickle.load(f)


parser = argparse.ArgumentParser(description='parsing arguments')
parser.add_argument('-class_idx', action="store",  dest="class_idx", type=int)
parser.add_argument('-ratio_valid', action="store",  dest="ratio_valid", type=float)
parser.add_argument('-k_fold', action="store",  dest="k_fold", type=int)
parser.add_argument('-use_model', action="store",  dest="use_model")
parser.add_argument('-create_new_data', action="store",  dest="create_new_data", type=int)
parser.add_argument('-level', action="store",  dest="level", type=int)
parser.add_argument('-n_k_fold', action="store",  dest="n_k_fold", type=int)
parser.add_argument('-n_threads', action="store",  dest="n_threads", type=int)
args = parser.parse_args()
class_idx = args.class_idx
ratio_valid = args.ratio_valid
k_fold = args.k_fold
use_model = args.use_model
create_new_data = bool(args.create_new_data)
level = args.level
n_k_fold = args.n_k_fold
n_threads = args.n_threads
assert create_new_data == False
get_prediction_kfold = True

n_iters = {}
if level==1:
	if class_idx==-1:
		n_iters['xgb'] = 6500
		n_iters['lightgbm'] = 4500
	else:
		n_iters['xgb'] = 1400
		n_iters['lightgbm'] = 1100
else:
	if class_idx==-1:
		n_iters['xgb'] = 450
		n_iters['lightgbm'] = 500
	else:
		n_iters['xgb'] = 500
		n_iters['lightgbm'] = 500
max_depth = 5


if __name__ == "__main__":
	# Load extracted feature-data files
	data_fname = 'data/extracted_features/data.class_idx.{}.{}_fold.{}.{}.lv{}.pickle'.format(class_idx, n_k_fold, k_fold, ratio_valid,level)
	data_train, data_valid, sku_id_train, sku_id_valid = objFromFile(data_fname)
	x_train = data_train[:,:-2]
	y_train = data_train[:,class_idx]

	# Model settings
	if use_model=='lightgbm':
		param = {
				'application': 'binary',
				'predict_raw_score': True,
				'boosting_type': 'gbdt',
				'metric': ['l2_root'],
				'num_threads': n_threads,
				'learning_rate': 0.01,
				'num_leaves': 20,
				'verbosity': 0,
				}
		d_train = lgb.Dataset(x_train, label=y_train)
		d_valid = lgb.Dataset(data_valid[:,:-2], label=data_valid[:,class_idx])
	elif use_model=='xgb':
		xgb = XGBClassifier(
			learning_rate = 0.01,
			n_estimators = n_iters[use_model],
			max_depth = 5,
			min_child_weight = 1,
			gamma = 0,
			subsample = 0.8, 
			colsample_bytree = 0.8,
			objective = 'binary:logistic',
			nthread = n_threads,
			reg_alpha = 0.005,
			scale_pos_weight = 1,
			seed = 123
		)

	# Training
	if ratio_valid>0:
		if use_model=='xgb':
			xgb.fit(x_train, y_train,
			early_stopping_rounds=250 if not get_prediction_kfold else 40000,
			eval_set=[(x_train, y_train),(data_valid[:,:-2], data_valid[:,class_idx])],
			eval_metric='rmse' #'rmse'
		)
		elif use_model=='lightgbm':
			bst = lgb.train(param, d_train, n_iters[use_model], valid_sets=(d_train,d_valid), early_stopping_rounds=100 if not get_prediction_kfold else 40000) # try alway use early stop
	else:
		if use_model=='xgb':
			xgb.fit(data_train[:,:-2], data_train[:,class_idx])
		elif use_model=='lightgbm':
			bst = lgb.train(param, d_train, n_iters[use_model])
		
	# Predicting
	if use_model=='xgb':
		predictions = xgb.predict_proba(data_valid[:,:-2])[:,1]
	elif use_model=='lightgbm':
		predictions = bst.predict(data_valid[:,:-2])
	y_golden = data_valid[:,class_idx]

	# Write k-fold predictions to file
	if get_prediction_kfold:
		fname = 'data/predictions/{}.{}.class_idx.{}.lv{}.pkl'.format(use_model,ratio_valid,class_idx,level)
		try:
			k_fold_predictions = objFromFile(fname)
		except:
			k_fold_predictions = {}
		for i,id in enumerate(sku_id_valid):
			k_fold_predictions[id] = predictions[i]
		objToFile(k_fold_predictions, fname)
		
	# print 'mean(predictions)', np.mean(predictions), 'mean(y_golden)', np.mean(y_golden)
	# print 'rmse(raw)', np.sqrt(np.mean((predictions-y_golden)**2))

	# Write test predections to file
	if ratio_valid==0:
		target = 'clarity' if class_idx==-2 else 'conciseness'
		res_file = 'to_submit/{}.lv{}.{}_test.predict'.format(use_model,level,target)
		with open(res_file, 'w') as f:
			for _ in predictions:
				f.write('{}\n'.format(_))



