#!/bin/sh

date  

python preprocess.py
python create_embeddings.py
cd brown-cluster-master 
./wcluster --text all_titles_lowercase.txt --c 60
./wcluster --text all_titles_lowercase.txt --c 80
./wcluster --text all_titles_lowercase.txt --c 100
./wcluster --text all_titles_lowercase.txt --c 120

date  
