'''
Pre-process the input datafile: empty fields handling, HTML converting, create header for csv files.
Create text input for Brown clustering

'''

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup


def normalize_text(text):
	text = re.sub(r'[0-9]+', ' X ', text)
	text = text.replace('-',' ')
	return text


def pre_process(df):
	df['short_description'].fillna("NIL_DESP", inplace=True)
	df['category_lvl_1'] = df['category_lvl_1'].fillna("NIL_C1")
	df['category_lvl_2'] = df['category_lvl_2'].fillna("NIL_C2")
	df['category_lvl_3'] = df['category_lvl_3'].fillna("NIL_C3")
	df['product_type'] = df['product_type'].fillna("NIL_PT")
	for c in df.columns.values:
		try:
			assert df[c].isnull().values.any()==False
		except:
			print c
			raise
	df['short_description'] = map(lambda x:' '.join((' '.join(BeautifulSoup(x).strings)).split()),df['short_description'])
	df['short_description'] = map(lambda x:x if len(x)>0 and x!='NULL' else "NIL_DESP",df['short_description'])

	# df['short_description'] = map(lambda x:' '.join(TweetTokenizer().tokenize(normalize_text(x))) if len(x)>0 and x!='NULL' else "NIL_DESP",df['short_description'])
	# df['short_description'] = map(lambda x:x if len(x)>0 and x!='NULL' else "NIL_DESP",df['short_description'])
	# df['title'] = map(lambda x: ' '.join(TweetTokenizer().tokenize(normalize_text(x))), df['title'])


if __name__ == "__main__":
	# Add headers
	headers = ["country","sku_id","title","category_lvl_1","category_lvl_2","category_lvl_3","short_description","price","product_type","clarity","conciseness"]

	# Training data file
	df = pd.read_csv("data/train/data_train.csv", delimiter=',', encoding='utf-8', names = headers)
	with open("data/train/clarity_train.labels",'r') as f:
		clarity = [int(_.strip()) for _ in f.readlines()]
	df['clarity'] = clarity
	with open("data/train/conciseness_train.labels",'r') as f:
		conciseness = [int(_.strip()) for _ in f.readlines()]
	df['conciseness'] = conciseness
	pre_process(df)
	df.to_csv('data/train/data_train_preprocessed.csv', ',', encoding='utf-8',index=False)

	# Test1 data file (for phase 1)
	df = pd.read_csv("data/test1/data_valid.csv", delimiter=',', encoding='utf-8', names = headers[:-2])
	pre_process(df)
	df.to_csv('data/test1/data_valid_preprocessed.csv', ',', encoding='utf-8',index=False)

	# Test2 data file (for phase 2)
	df = pd.read_csv("data/test2/data_test.csv", delimiter=',', encoding='utf-8', names = headers[:-2])
	pre_process(df)
	df.to_csv('data/test2/data_test_preprocessed.csv', ',', encoding='utf-8',index=False)

	# Generate text input for Brown clustering
	df_train_all = pd.read_csv("data/train/data_train_preprocessed.csv",delimiter=',', encoding='utf-8')
	df_test = pd.read_csv("data/test1/data_valid_preprocessed.csv",delimiter=',', encoding='utf-8')
	df_test2 = pd.read_csv("data/test2/data_test_preprocessed.csv",delimiter=',', encoding='utf-8')
	with open('brown-cluster-master/all_titles_lowercase.txt','w') as fo:
		for _ in np.hstack((df_train_all['title'].values,df_test['title'].values,df_test2['title'].values)):
			fo.write('{}\n'.format(_.strip().lower().encode('utf8')))