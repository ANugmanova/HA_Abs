#!/bin/bash

# export fasta file
python ./mBLM/script/df2fasta.py ./mBLM/result/OAS_memory_paired_v3.csv ./data/OAS_memory_paired_clean.fasta

# clustering
output_prefix=./mBLM/result/cluster/
f=./data/OAS_memory_paired_clean.fasta
name=OAS_memory_paired_clean
for x in 0.5 0.6
do
	cd-hit -i $f -o $output_prefix$name$x -c $x -M 32000 -d 0 -T 32 -n 3 -aL 0.8 -s 0.95  -uS 0.2  -sc 1 -sf 1
done
    
for x in 0.7 0.8 0.9
do
	cd-hit -i $f -o $output_prefix$name$x -c $x -M 32000 -d 0 -T 32 -n 5 -aL 0.8 -s 0.95  -uS 0.2  -sc 1 -sf 1
done  


# add cluster id to dataset

python ./mBLM/script/add_clstr2df.py -i ./mBLM/result/OAS_memory_paired_v3.csv -o ./mBLM/result/OAS_memory_paired_v4.csv

# split dataset
python ./mBLM/script/split_dataset.py -i ./mBLM/result/OAS_memory_paired_v4.csv -o ./data/dataset/OAS_memory
