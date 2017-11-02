import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
import argparse


def objToFile(obj, path):
	print "Writing to {}".format(path)
	with open(path,'wb') as f:
			pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def objFromFile(path):
	with open(path, 'rb') as f:
		return pickle.load(f)


def evaluate(f,target, target_cat=None):
	df_train_all = pd.read_csv("data/train/data_train_preprocessed.csv",\
		delimiter=',', encoding='utf-8')
	s = 0.
	n =1.
	for i,r in df_train_all.iterrows():
		if target_cat==None or r['category_lvl_1'] == target_cat:
			s+=(r[target] - f[r['sku_id']]) * (r[target] - f[r['sku_id']])
			n+=1.
	print np.sqrt(s/n),'' if target_cat == None else '{}({})'.format(target_cat, int(n))


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='parsing arguments')
	parser.add_argument('-class_idx', action="store", dest="class_idx", type=int)
	parser.add_argument('-use_model', action="store", dest="use_model")
	parser.add_argument('-level', action="store", dest="level", type=int)
	args = parser.parse_args()
	class_idx = args.class_idx
	use_model = args.use_model
	level = args.level

	f1 = objFromFile('data/predictions/{}.0.2.class_idx.{}.lv{}.pkl'.format(use_model, class_idx, level))

	print ('RMSE:')
	evaluate(f1, 'clarity' if class_idx == -2 else 'conciseness', target_cat=None)

