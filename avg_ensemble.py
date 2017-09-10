'''
Average ensemble
'''

import numpy as np


if __name__ == "__main__":
	for class_idx in [-2,-1]:
		target = 'clarity' if class_idx==-2 else 'conciseness'
		models_predictions = []
		for use_model in ['xgb.lv2','lightgbm.lv2'] if class_idx==-2 else ['xgb.lv2','xgb.lv2']:
			res_file = 'to_submit/{}.{}_test.predict'.format(use_model,target)
			with open(res_file) as f:
				predictions = map(lambda x: float(x.strip()), f.readlines())
				models_predictions.append(predictions)
		
		avg_predictions = np.mean(models_predictions, axis=0)

		res_file = 'to_submit/ensemble.lv2.{}_test.predict'.format(target)
		with open(res_file, 'w') as f:
			for _ in avg_predictions:
				f.write('{}\n'.format(_))