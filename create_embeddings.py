'''
Create word2vec embeddings from the given corpus (titles and descriptions)

'''

import numpy as np
import pandas as pd
from tqdm import tqdm
import gensim
np.random.seed(1)


if __name__ == "__main__":
	
	df_train = pd.read_csv("data/train/data_train_preprocessed.csv",delimiter=',', encoding='utf-8')
	df_test = pd.read_csv("data/test1/data_valid_preprocessed.csv",delimiter=',', encoding='utf-8')
	df_test2 = pd.read_csv("data/test2/data_test_preprocessed.csv",delimiter=',', encoding='utf-8')

	sentences = [sen.lower().split() for sen in df_train['title'].values] + [sen.lower().split() for sen in df_train['short_description'].values]
	sentences += [sen.lower().split() for sen in df_test['title'].values] + [sen.lower().split() for sen in df_test['short_description'].values]
	sentences += [sen.lower().split() for sen in df_test2['title'].values] + [sen.lower().split() for sen in df_test2['short_description'].values]
	np.random.shuffle(sentences)

	model = gensim.models.Word2Vec(sentences, sg=1, size=100, window=5, min_count=3, workers=8)
	try:
		model.save_word2vec_format('data/w2v.txt', binary=False)
	except:
		model.wv.save_word2vec_format('data/w2v.txt', binary=False)