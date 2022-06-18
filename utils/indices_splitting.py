# Must have /temp/ with raw data

TRAIN_FILE = 'train_idx_balanced.csv' 
VAL_FILE = 'val_idx_balanced.csv' 
TEST_FILE = 'test_idx_balanced.csv' 

import pandas as pd
import numpy as np
import os
import shutil

source = 'raw_indices/'
outdir = 'new_indices'

if not outdir in os.listdir():
	os.mkdir(outdir)

part2filename = {
	'Train': TRAIN_FILE,
	'Val': VAL_FILE,
	'Test': TEST_FILE,
}

for partition in ['Train', 'Val', 'Test']:
	filename = part2filename[partition]
	df = pd.read_csv(source + filename, names=range(1,11), header=0); 
	#df.head()
	for i in range(1, 11):
		df_np = df[i].to_csv(f'{outdir}/i{partition}B{i}.csv', index=False)

for filename in os.listdir(outdir):
	shutil.move(outdir + '/' + filename, '../data/partitions/csv/')

os.rmdir(outdir)


































