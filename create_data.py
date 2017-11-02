'''
Extract features and create training/testing data for k-fold validation
'''

import argparse, pickle, itertools
from tqdm import tqdm
import numpy as np
import scipy
from numpy import sort
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk, spacy, re
from collections import Counter
from collections import defaultdict
from sklearn.model_selection import KFold
from sklearn.decomposition import NMF, LatentDirichletAllocation


def normalize_text(text):
	text = re.sub(r'[0-9]+', ' X ', text)
	text = text.replace('-', ' ')
	return text


def objToFile(obj, path):
	print "Writing to {}".format(path)
	with open(path, 'wb') as f:
			pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def objFromFile(path):
	with open(path, 'rb') as f:
		return pickle.load(f)


def brown_sim(p1, p2):
	if p1 == p2:
		return 1.0
	else:
		l = 0
		while p1[l] == p2[l] and l < len(p1) and l < len(p2):
			l += 1
		return float(l) / max(len(p1), len(p2))


def jaccard_sim(s1, s2):
	return float(len(s1&s2)) / len(s1|s2) if len(s1|s2) > 0 else -1.0


def pos_count2features(lst, pos2id):
	counter = Counter(lst)
	tmp = [0.0] * len(pos2id)
	for _ in counter.keys():
		try:
			tmp[pos2id[_]] = counter[_]
		except:
			continue
	return tmp


def sims2features(sims, min_range=0.0, max_range=1.0, n_bins=5):
	res = []
	res.append(len(sims))
	if n_bins == 0:
		if len(sims)>0:
			res += [min(sims), max(sims), np.mean(sims)]
		else:
			res += [-1] * 3
	else:
		if len(sims) > 0:
			if min_range == None:
				min_range = min(sims)
			if max_range == None:
				max_range = max(sims)
			org_hist = np.histogram(sims, bins = n_bins,\
							range=(min_range, max_range))[0]
			scaled_hist = org_hist.astype(np.float32) / sum(org_hist)
			res += list(org_hist) + list(scaled_hist) \
					+ [min(sims), max(sims), np.mean(sims)]
		else:
			res += [-1] * (n_bins * 2 + 3)
	return res


def get_features(df):
	df_cat = df[cat_cols].values
	data = []
	for i,r in tqdm(df.iterrows()):
		fs = []
		title_lw = r['title'].lower()
		ws_title = r['title'].lower().split()
		ws_desc = r['short_description'].lower().split()
		desc_lw = r['short_description'].lower()
		ws_title_org = r['title'].split()
		ws_desc_org = r['short_description'].split()
		sku_id = r['sku_id']

		fs.append(len(r['title']))
		fs.append(len(r['title']) / len(ws_title))
		fs.append(len(ws_title))
		fs.append(len(set(ws_title)))
		fs.append(len(ws_title) - len(set(ws_title)))
		fs.append(float(len(ws_title) - len(set(ws_title))) / len(ws_title))

		# Popularity of words
		if use_word_popularity:
			p = []
			for w in ws_title:
				try:
					p.append(np.log(p_words[w] + 1.0))
				except:
					continue
			fs += sims2features(p, min_range=None, max_range=None, n_bins=0)

			# Word popularity of desc
			p = []
			for w in ws_desc:
				try:
					p.append(np.log(p_words[w] + 1.0))
				except:
					continue
			fs + =sims2features(p, min_range=None, max_range=None, n_bins=0)

		fs.append(sku_id[:2].lower() == title_lw[:2])
		fs.append(any(w.isdigit() for w in ws_title))
		fs.append(any(c.isdigit() for c in r['title']))
		fs.append(np.sum([int(w.isdigit()) for w in ws_title]))
		fs.append(np.sum([int(c.isdigit()) for c in r['title']]))

		for w in top_words:
			fs.append(title_lw.count(w))

		for w in last_words:
			fs.append(ws_title[-1] == w)

		if use_neg_word_features:
			for w in top_neg_words:
				fs.append(title_lw.count(w))
			fs.append(sum([int(_ in top_neg_words_set) for _ in ws_title]))
			fs.append(len(set(ws_title) & top_neg_words_set))

		fs.append(len(ws_title_org) > 0 and ws_title_org[0].isupper())
		if class_idx == -1:
			n_upper_words = sum([w.isupper() for w in ws_title_org])
			fs.append(n_upper_words)
			fs.append(float(n_upper_words)/len(ws_title_org))
			fs.append(np.sum([int(w.isupper()) for w in ws_title_org])\
												  / len(ws_title_org))

		fs.append(r['title'].isupper())
		fs.append(len(ws_desc))
		fs.append(float(len(ws_desc)) / len(ws_title))
		fs.append(len(ws_desc) - len(ws_title))
		fs.append(len(set(ws_title) & set(ws_desc)))
		fs.append(jaccard_sim(set(ws_title), set(ws_desc)))

		fs.append(any(w.isdigit() for w in ws_desc))
		fs.append(any(c.isdigit() for c in r['short_description']))
		fs.append(r['short_description'].isupper())
		if class_idx == -1:
			fs.append(sum([w.isupper() for w in ws_desc_org]))
			fs.append(np.sum([int(w.isupper()) for w in ws_desc_org])\
													/len(ws_desc_org))

		fs.append(jaccard_sim(set([w for w in ws_desc if w.isdigit()]),\
							set([w for w in ws_title if w.isdigit()])))
		fs.append(jaccard_sim(set(ws_title),set(ws_desc))) 

		cat1 = r['category_lvl_1'].lower().split()
		cat2 = r['category_lvl_2'].lower().split()
		cat3 = r['category_lvl_3'].lower().split()
		fs.append(len(set(ws_title)&set(cat1)))
		fs.append(len(set(ws_title)&set(cat2)))
		fs.append(len(set(ws_title)&set(cat3)))

		fs.append(r['price'])
		fs.append(r['price'] * price_rate[r['country']])

		fs+=df_cat[i].tolist()

		if use_noun_chunk_features:
			doc = nlp(r['title'])
			nchunks = [_ for _ in doc.noun_chunks]
			fs.append(len(nchunks))
			len_nchunks = [len(_.text.split()) for _ in nchunks]
			s_noun = sum(len_nchunks)
			fs.append(s_noun)
			fs.append(len(ws_title) - s_noun)
			len_nchunks = sorted(len_nchunks, reverse=True)
			fs+=len_nchunks[:3] + [-1] * (3-len(len_nchunks))
			fs+=pos_count2features([token.pos for token in doc], spacy_pos2id)
		fs+=pos_count2features([_[1] for _ in nltk.pos_tag(r['title'].lower().split())], nltk_pos2id)
		fs+=pos_count2features([token.tag_ for token in nlp(r['title'].lower())], nltk_pos2id)

		# Word_brown features
		for _, word_brown in enumerate(word_browns):
			ws_br = []
			for w in r['title'].lower().split():
				try:
					ws_br.append(word_brown[w])
				except:
					continue
			fs += pos_count2features(ws_br, brown_clusters2id[_])

			max_sim = max_sim2 = -2
			min_sim = min_sim2 = 2
			sims = []
			k_sims = 3
			for (v1,v2) in itertools.combinations(ws_br, 2):
				cos_sim = brown_sim(v1, v2)
				max_sim = max(max_sim, cos_sim)
				min_sim = min(min_sim, cos_sim)
				if cos_sim < 0.9999:
					max_sim2 = max(max_sim, cos_sim)
					min_sim2 = min(min_sim, cos_sim)
				sims.append(cos_sim)
			fs += sims2features(sims, min_range=None, max_range=None, n_bins=0)
			fs.append(len(sims))
			n_sims_zero = np.sum((np.array(sims)==1).astype(np.int32))
			fs.append(n_sims_zero)
			if  len(sims)>0:
				sims = sorted(sims)
				fs += sims[:k_sims] + [-1] * (k_sims-len(sims[:k_sims]))
				fs += sims[-k_sims:] + [-1] * (k_sims-len(sims[-k_sims:]))
			else:
				fs += [-2] * 6
			fs += [max_sim, max_sim2, min_sim, min_sim2]

		if use_emb_features:
			k_sims = 3
			for emb_vec,emb_vocab in emb_vecs:
				ws_embs = [emb_vec[_] for _ in ws_title if _ in emb_vocab]
				fs.append(len(set(ws_title)&emb_vocab))
				fs.append(len(ws_embs))
				fs.append(float(len(ws_embs))/len(ws_title))
				max_sim = max_sim2 = -2
				min_sim = min_sim2 = 2
				sims = []
				for (v1,v2) in itertools.combinations(ws_embs,2):
					cos_sim = 1.0-scipy.spatial.distance.cosine(v1,v2)
					max_sim = max(max_sim, cos_sim)
					min_sim = min(min_sim, cos_sim)
					if cos_sim<0.9999:
						max_sim2 = max(max_sim, cos_sim)
						min_sim2 = min(min_sim, cos_sim)
					sims.append(cos_sim)
				fs.append(len(sims))
				if  len(sims) > 0:
					sims = sorted(sims)
					fs += sims[:k_sims] + [-1] * (k_sims-len(sims[:k_sims]))
					fs += sims[-k_sims:] + [-1] * (k_sims-len(sims[-k_sims:]))
				else:
					fs += [-2]*(2*k_sims)
				fs += [max_sim, max_sim2, min_sim, min_sim2]

		if use_k_fold_features:
			for _ in k_fold_predictions + predictions_first_digit_lst\
											 + predictions_order_lst:
				try:
					fs.append(_[sku_id])
				except:
					continue

		if use_topic_model:
			for _ in title_embs:
				fs += list(_[sku_id])

		if class_idx == -1:
			c = r['category_lvl_1'] + r['category_lvl_2'] + r['category_lvl_3']
			p = float(r['price'])
			try:
				fs.append(p / max_cat_prices[c])
				fs.append(p - mean_cat_prices[c])
			except:
				fs += [-1] * 2

		fs.append(r['clarity'])
		fs.append(r['conciseness'])

		data.append(fs)
	return np.array(data)	


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='parsing arguments')
	parser.add_argument('-class_idx', action="store", dest="class_idx", type=int)
	parser.add_argument('-ratio_valid', action="store", dest="ratio_valid", type=float)
	parser.add_argument('-k_fold', action="store",  dest="k_fold", type=int)
	parser.add_argument('-use_model', action="store", dest="use_model")
	parser.add_argument('-create_new_data', action="store", dest="create_new_data", type=int)
	parser.add_argument('-level', action="store", dest="level", type=int)
	parser.add_argument('-n_k_fold', action="store", dest="n_k_fold", type=int)
	parser.add_argument('-over_sampling', action="store", dest="over_sampling", type=float)
	args = parser.parse_args()
	class_idx = args.class_idx
	ratio_valid = args.ratio_valid
	k_fold = args.k_fold
	use_model = args.use_model
	create_new_data = bool(args.create_new_data)
	level = args.level
	n_k_fold = args.n_k_fold
	over_sampling = args.over_sampling


	if level == 2:
		use_neg_word_features = False
		use_emb_features = False
		use_noun_chunk_features = False
		use_k_fold_features = True
		get_prediction_kfold = True
		use_topic_model = class_idx==-1
	else:
		use_neg_word_features = class_idx==-1
		use_emb_features = True
		use_noun_chunk_features = True
		use_k_fold_features = False
		get_prediction_kfold = True
		use_topic_model = class_idx==-2
	use_word_popularity = class_idx==-2

	np.random.seed(-class_idx)
	nlp = spacy.load('en')


	if create_new_data:
		df_train_all = pd.read_csv("data/train/data_train_preprocessed.csv",\
											delimiter=',', encoding='utf-8')
		df_test = pd.read_csv("data/test2/data_test_preprocessed.csv",\
										delimiter=',', encoding='utf-8')
		df_train_all = df_train_all.sample(frac=1).reset_index(drop=True)
		if ratio_valid > 0:
			if not get_prediction_kfold:
				split_point = int(ratio_valid * df_train_all.shape[0])
				df_valid = df_train_all[-split_point:]
				df_valid.reset_index(inplace=True,drop=True)
				df_train = df_train_all[:-split_point]
			else:
				if class_idx == -2:
					df_train_all = df_train_all.sample(frac=1).reset_index(drop=True)
				kf = KFold(n_splits=n_k_fold)
				i = 0
				for train_index, test_index in kf.split(range(len(df_train_all))):
					if k_fold == i:
						test_index = np.array(test_index)
						train_index = np.array(train_index)
						df_valid = df_train_all.iloc[test_index]
						df_train = df_train_all.iloc[train_index]
						df_train.reset_index(inplace=True,drop=True)
						df_valid.reset_index(inplace=True,drop=True)
					i += 1
		else:
			df_train = df_train_all
			df_valid = df_test
			df_valid['clarity'] = 0
			df_valid['conciseness'] = 0

		if use_emb_features:
			emb_vecs = []
			sing_word_vocab = set((' '.join(df_train['title'].values) + ' '\
					+ ' '.join(df_valid['title'].values)).lower().split())
			for fname in ['w2v.txt','numberbatch-en.txt']:
				f = open('data/{}'.format(fname))
				print 'Reading pre-trained emb file'
				emb_vec = {}
				for line in tqdm(f):
					values = line.split()
					if len(values) == 2:
						continue
					word = values[0]
					if word not in sing_word_vocab:
						continue
					coefs = np.asarray(values[1:], dtype='float32')
					emb_vec[word] = coefs
				print fname, len(sing_word_vocab), 'single_words.',\
									 len(emb_vec), 'words have emb'
				emb_vecs.append((emb_vec,set(emb_vec.keys())))
		
		if use_noun_chunk_features:
			# Get pos dict
			all_pos_ = []
			print 'get spacy_pos2id'
			for i,r in tqdm(df_train.iterrows()):
				doc = nlp(r['title'])
				for token in doc:
					all_pos_.append(token.pos)
			all_pos_ = list(set(all_pos_))
			spacy_pos2id = dict(zip(all_pos_, range(len(all_pos_))))
			print 'len(spacy_pos2id) = ', len(spacy_pos2id), all_pos_

		n_top_words = 1500 if class_idx==-2 else 1000
		top_words = [_[0] for _ in Counter(' '.join(df_train['title'].values).lower().split()).most_common(n_top_words)]
		print 'top common words', top_words[:100]

		n_last_words = 500
		last_words = [_[0] for _ in Counter([title.strip().lower().split()[-1] for title in df_train['title'].values]).most_common(n_last_words)]
		print 'top last words', last_words[:100]

		# Popularity of words
		if use_word_popularity:
			p_words  = Counter(' '.join(df_train['title'].values).lower().split())

		if use_neg_word_features:
			n_top_neg_words = 700
			top_neg_words1 = [_[0] for _ in Counter((' '.join(df_train.loc[df_train['clarity'] == 0]['title'].values)).lower().split()).most_common(n_top_neg_words)]
			top_neg_words2 = [_[0] for _ in Counter((' '.join(df_train.loc[df_train['conciseness'] == 0]['title'].values)).lower().split()).most_common(n_top_neg_words)]
			top_neg_words = list(set(top_neg_words1 + top_neg_words2))
			print 'top negative words', top_neg_words[:100]
			print 'common words intersect', len(set(top_neg_words) & set(top_words))
			top_neg_words = list(set(top_neg_words) - set(top_words))
			print len(top_neg_words), 'distinct neg words left'
			top_neg_words_set = set(top_neg_words)

		special_chars = '+-=&;,.(-/?[*,$:'
		price_rate = {'ph':0.020, 'my':0.23, 'sg':0.71}

		cat_cols = []
		df = pd.concat([df_train,df_valid])
		for feature in ['country','category_lvl_1','category_lvl_2',\
									'category_lvl_3','product_type']:
			dummies = pd.get_dummies(df[feature], prefix='{}.'.format(feature), sparse=True)
			for col in dummies.columns:
				df_train[col] = dummies[col][:len(df_train)]
				df_valid[col] = dummies[col][-len(df_valid):]
				cat_cols.append(col)

		#Brow clustering distance
		word_browns = []
		for fname in ['all_titles_lowercase-c120-p1.out',\
						'all_titles_lowercase-c100-p1.out',\
						'all_titles_lowercase-c80-p1.out',\
						'all_titles_lowercase-c60-p1.out']:
			word_brown = {}
			with open('brown-cluster-master/{}/paths'.format(fname)) as f:
				for line in f:
					path, w, freq = line.strip().split()
					word_brown[unicode(w,'utf8')] = path
			word_browns.append(word_brown)
		brown_clusters = [list(set(_.values())) for _ in word_browns]
		brown_clusters2id = [dict(zip(_, range(len(_)))) for _ in brown_clusters]
		print 'len(brown_clusters)', [len(_) for _ in brown_clusters]

		print 'get nltk_pos2id'
		all_pos_=['PRP$', 'VBG', 'VBD', 'VBN',  'VBP', 'WDT', 'JJ', 'WP',\
				 'VBZ', 'DT', 'RP',  'NN', 'FW', 'POS',  'TO', 'LS', 'RB',\
				 'NNS', 'NNP', 'VB', 'WRB', 'CC', 'PDT', 'RBS', 'RBR',\
				  'CD', 'PRP', 'EX', 'IN', 'WP$', 'MD', 'NNPS', 'JJS', 'JJR',\
				   'SYM', 'UH']
		nltk_pos2id = dict(zip(all_pos_, range(len(all_pos_))))
		print 'len(nltk_pos2id) = ', len(nltk_pos2id)

		if use_k_fold_features:
			fnames = ['data/predictions/xgb.0.2.class_idx.-1.lv1.pkl',\
			'data/predictions/xgb.0.0.class_idx.-1.lv1.pkl',\
			'data/predictions/xgb.0.2.class_idx.-2.lv1.pkl',\
			'data/predictions/xgb.0.0.class_idx.-2.lv1.pkl']
			k_fold_predictions = [objFromFile(_) for _ in fnames]

			predictions_first_digit_lst = []
			predictions_order_lst = []
			for predictions in k_fold_predictions:
				predictions_first_digit = {}
				for suid in predictions.keys():
					predictions_first_digit[suid] = int(predictions[suid]*10)
				predictions_first_digit_lst.append(predictions_first_digit)

				predictions_order = {}
				bin_size = len(predictions)/10
				for i,_ in enumerate(sorted(list(predictions.items()), key=lambda x:x[-1])):
					predictions_order[_[0]] = int(i/bin_size)
				predictions_order_lst.append(predictions_order)

		prior_conciseness_by_cat_lst = defaultdict(list)
		prior_clarity_by_cat_lst = defaultdict(list)
		for _ in zip(df_train['category_lvl_1'].values, df_train['conciseness'].values,\
					 df_train['clarity'].values):
			prior_conciseness_by_cat_lst[_[0]].append(_[1])
			prior_clarity_by_cat_lst[_[0]].append(_[2])

		if use_topic_model:
			normal_tokenizer = lambda x:x.lower().split()
			documents = np.hstack((df_train['title'].values,\
									df_valid['title'].values))
			documents_ids = np.hstack((df_train['sku_id'].values,\
										 df_valid['sku_id'].values))
			# NMF is able to use tf-idf
			tfidf_vectorizer = TfidfVectorizer(min_df=2, tokenizer=normal_tokenizer)
			tfidf = tfidf_vectorizer.fit_transform(documents)

			tf_vectorizer = TfidfVectorizer(min_df=2, tokenizer=normal_tokenizer)
			tf = tf_vectorizer.fit_transform(documents)

			no_topics = 50
			# Run NMF
			nmf = NMF(n_components=no_topics, random_state=1, alpha=.1,\
						 l1_ratio=.5, init='nndsvd').fit_transform(tfidf)
			nmf2vec = {}
			for i in range(len(documents_ids)):
				nmf2vec[documents_ids[i]] = nmf[i]

			# Run LDA
			lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5,\
							 learning_method='online', learning_offset=50.,\
							 random_state=0).fit_transform(tf)
			lda2vec = {}
			for i in range(len(documents_ids)):
				lda2vec[documents_ids[i]] = lda[i]

			title_embs = [nmf2vec, lda2vec]

		cat_prices = defaultdict(list)
		for i,r in df_train.iterrows():
			c = r['category_lvl_1']+r['category_lvl_2']+r['category_lvl_3']
			cat_prices[c].append(float(r['price']))
		max_cat_prices = {}
		avg_cat_prices = {}
		for c in cat_prices.keys():
			max_cat_prices[c] = np.max(cat_prices[c])
			avg_cat_prices[c] = np.mean(cat_prices[c])


	# Write feature data into file
	if create_new_data:
		data_fname = 'data/extracted_features/data.class_idx\
						.{}.{}_fold.{}.{}.lv{}.pickle'\
						.format(class_idx, n_k_fold,\
							k_fold, ratio_valid,level)
		data_train = get_features(df_train) 
		data_valid = get_features(df_valid)
		if over_sampling>0:
			data_train_neg = np.array([_ for _ in data_train if _[class_idx]==0])
			np.random.shuffle(data_train_neg)
			data_train = np.vstack((data_train, \
				data_train_neg[:int(len(data_train_neg) * over_sampling)]))
			np.random.shuffle(data_train)
		sku_id_train = df_train['sku_id'].values
		sku_id_valid = df_valid['sku_id'].values
		objToFile([data_train, data_valid,sku_id_train,sku_id_valid],data_fname)
	else:
		data_train, data_valid, sku_id_train, sku_id_valid = objFromFile(data_fname)