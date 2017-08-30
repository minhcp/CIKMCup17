#!/bin/sh

date  

mkdir data/predictions
mkdir data/extracted_features
mkdir to_submit

python create_data.py -class_idx -1 -ratio_valid 0.0 -k_fold -1 -use_model xgb -create_new_data 1 -level 1 -n_k_fold 10 -over_sampling 0.2&
python create_data.py -class_idx -1 -ratio_valid 0.2 -k_fold 0 -use_model xgb -create_new_data 1 -level 1 -n_k_fold 10 -over_sampling 0.2&
python create_data.py -class_idx -1 -ratio_valid 0.2 -k_fold 1 -use_model xgb -create_new_data 1 -level 1 -n_k_fold 10 -over_sampling 0.2&
python create_data.py -class_idx -1 -ratio_valid 0.2 -k_fold 2 -use_model xgb -create_new_data 1 -level 1 -n_k_fold 10 -over_sampling 0.2&
python create_data.py -class_idx -1 -ratio_valid 0.2 -k_fold 3 -use_model xgb -create_new_data 1 -level 1 -n_k_fold 10 -over_sampling 0.2&
python create_data.py -class_idx -1 -ratio_valid 0.2 -k_fold 4 -use_model xgb -create_new_data 1 -level 1 -n_k_fold 10 -over_sampling 0.2&
python create_data.py -class_idx -1 -ratio_valid 0.2 -k_fold 5 -use_model xgb -create_new_data 1 -level 1 -n_k_fold 10 -over_sampling 0.2&
python create_data.py -class_idx -1 -ratio_valid 0.2 -k_fold 6 -use_model xgb -create_new_data 1 -level 1 -n_k_fold 10 -over_sampling 0.2&
python create_data.py -class_idx -1 -ratio_valid 0.2 -k_fold 7 -use_model xgb -create_new_data 1 -level 1 -n_k_fold 10 -over_sampling 0.2&
python create_data.py -class_idx -1 -ratio_valid 0.2 -k_fold 8 -use_model xgb -create_new_data 1 -level 1 -n_k_fold 10 -over_sampling 0.2&
python create_data.py -class_idx -1 -ratio_valid 0.2 -k_fold 9 -use_model xgb -create_new_data 1 -level 1 -n_k_fold 10 -over_sampling 0.2&
wait

python create_data.py -class_idx -2 -ratio_valid 0.0 -k_fold -1 -use_model xgb -create_new_data 1 -level 1 -n_k_fold 10 -over_sampling 0.2&
python create_data.py -class_idx -2 -ratio_valid 0.2 -k_fold 0 -use_model xgb -create_new_data 1 -level 1 -n_k_fold 10 -over_sampling 0.2&
python create_data.py -class_idx -2 -ratio_valid 0.2 -k_fold 1 -use_model xgb -create_new_data 1 -level 1 -n_k_fold 10 -over_sampling 0.2&
python create_data.py -class_idx -2 -ratio_valid 0.2 -k_fold 2 -use_model xgb -create_new_data 1 -level 1 -n_k_fold 10 -over_sampling 0.2&
python create_data.py -class_idx -2 -ratio_valid 0.2 -k_fold 3 -use_model xgb -create_new_data 1 -level 1 -n_k_fold 10 -over_sampling 0.2&
python create_data.py -class_idx -2 -ratio_valid 0.2 -k_fold 4 -use_model xgb -create_new_data 1 -level 1 -n_k_fold 10 -over_sampling 0.2&
python create_data.py -class_idx -2 -ratio_valid 0.2 -k_fold 5 -use_model xgb -create_new_data 1 -level 1 -n_k_fold 10 -over_sampling 0.2&
python create_data.py -class_idx -2 -ratio_valid 0.2 -k_fold 6 -use_model xgb -create_new_data 1 -level 1 -n_k_fold 10 -over_sampling 0.2&
python create_data.py -class_idx -2 -ratio_valid 0.2 -k_fold 7 -use_model xgb -create_new_data 1 -level 1 -n_k_fold 10 -over_sampling 0.2&
python create_data.py -class_idx -2 -ratio_valid 0.2 -k_fold 8 -use_model xgb -create_new_data 1 -level 1 -n_k_fold 10 -over_sampling 0.2&
python create_data.py -class_idx -2 -ratio_valid 0.2 -k_fold 9 -use_model xgb -create_new_data 1 -level 1 -n_k_fold 10 -over_sampling 0.2&
wait


python predict.py -class_idx -1 -ratio_valid 0.2 -k_fold 0 -use_model xgb -create_new_data 0 -level 1 -n_k_fold 10 -n_threads 40
python predict.py -class_idx -1 -ratio_valid 0.2 -k_fold 1 -use_model xgb -create_new_data 0 -level 1 -n_k_fold 10 -n_threads 40
python predict.py -class_idx -1 -ratio_valid 0.2 -k_fold 2 -use_model xgb -create_new_data 0 -level 1 -n_k_fold 10 -n_threads 40
python predict.py -class_idx -1 -ratio_valid 0.2 -k_fold 3 -use_model xgb -create_new_data 0 -level 1 -n_k_fold 10 -n_threads 40
python predict.py -class_idx -1 -ratio_valid 0.2 -k_fold 4 -use_model xgb -create_new_data 0 -level 1 -n_k_fold 10 -n_threads 40
python predict.py -class_idx -1 -ratio_valid 0.2 -k_fold 5 -use_model xgb -create_new_data 0 -level 1 -n_k_fold 10 -n_threads 40
python predict.py -class_idx -1 -ratio_valid 0.2 -k_fold 6 -use_model xgb -create_new_data 0 -level 1 -n_k_fold 10 -n_threads 40
python predict.py -class_idx -1 -ratio_valid 0.2 -k_fold 7 -use_model xgb -create_new_data 0 -level 1 -n_k_fold 10 -n_threads 40
python predict.py -class_idx -1 -ratio_valid 0.2 -k_fold 8 -use_model xgb -create_new_data 0 -level 1 -n_k_fold 10 -n_threads 40
python predict.py -class_idx -1 -ratio_valid 0.2 -k_fold 9 -use_model xgb -create_new_data 0 -level 1 -n_k_fold 10 -n_threads 40
python evaluate.py -use_model xgb -class_idx -1 -level 1

python predict.py -class_idx -2 -ratio_valid 0.2 -k_fold 0 -use_model xgb -create_new_data 0 -level 1 -n_k_fold 10 -n_threads 40
python predict.py -class_idx -2 -ratio_valid 0.2 -k_fold 1 -use_model xgb -create_new_data 0 -level 1 -n_k_fold 10 -n_threads 40
python predict.py -class_idx -2 -ratio_valid 0.2 -k_fold 2 -use_model xgb -create_new_data 0 -level 1 -n_k_fold 10 -n_threads 40
python predict.py -class_idx -2 -ratio_valid 0.2 -k_fold 3 -use_model xgb -create_new_data 0 -level 1 -n_k_fold 10 -n_threads 40
python predict.py -class_idx -2 -ratio_valid 0.2 -k_fold 4 -use_model xgb -create_new_data 0 -level 1 -n_k_fold 10 -n_threads 40
python predict.py -class_idx -2 -ratio_valid 0.2 -k_fold 5 -use_model xgb -create_new_data 0 -level 1 -n_k_fold 10 -n_threads 40
python predict.py -class_idx -2 -ratio_valid 0.2 -k_fold 6 -use_model xgb -create_new_data 0 -level 1 -n_k_fold 10 -n_threads 40
python predict.py -class_idx -2 -ratio_valid 0.2 -k_fold 7 -use_model xgb -create_new_data 0 -level 1 -n_k_fold 10 -n_threads 40
python predict.py -class_idx -2 -ratio_valid 0.2 -k_fold 8 -use_model xgb -create_new_data 0 -level 1 -n_k_fold 10 -n_threads 40
python predict.py -class_idx -2 -ratio_valid 0.2 -k_fold 9 -use_model xgb -create_new_data 0 -level 1 -n_k_fold 10 -n_threads 40
python evaluate.py -use_model xgb -class_idx -2 -level 1

python predict.py -class_idx -1 -ratio_valid 0.0 -k_fold -1 -use_model xgb -create_new_data 0 -level 1 -n_k_fold 10 -n_threads 40
python predict.py -class_idx -2 -ratio_valid 0.0 -k_fold -1 -use_model xgb -create_new_data 0 -level 1 -n_k_fold 10 -n_threads 40

python create_data.py -class_idx -1 -ratio_valid 0.0 -k_fold -1 -use_model xgb -create_new_data 1 -level 2 -n_k_fold 10 -over_sampling 0.2&
python create_data.py -class_idx -2 -ratio_valid 0.0 -k_fold -1 -use_model xgb -create_new_data 1 -level 2 -n_k_fold 10 -over_sampling 0.2&
wait

python predict.py -class_idx -1 -ratio_valid 0.0 -k_fold -1 -use_model xgb -create_new_data 0 -level 2 -n_k_fold 10 -n_threads 40
python predict.py -class_idx -2 -ratio_valid 0.0 -k_fold -1 -use_model xgb -create_new_data 0 -level 2 -n_k_fold 10 -n_threads 40
python predict.py -class_idx -1 -ratio_valid 0.0 -k_fold -1 -use_model lightgbm -create_new_data 0 -level 2 -n_k_fold 10 -n_threads 40
python predict.py -class_idx -2 -ratio_valid 0.0 -k_fold -1 -use_model lightgbm -create_new_data 0 -level 2 -n_k_fold 10 -n_threads 40
python avg_ensemble.py

date  