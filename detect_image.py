import pdb
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
#%matplotlib inline

from os.path import join, dirname, abspath
from os import system


ROOT = dirname(abspath(__file__))

def detect(image_file, output_filename,caffemodel='models/bvlc_reference_rcnn_ilsvrc13/bvlc_reference_rcnn_ilsvrc13.caffemodel',deploy_prototxt='models/bvlc_reference_rcnn_ilsvrc13/deploy.prototxt'):
	cmd = join(ROOT, 'python/detect.py')
	cmd += ' --pretrained_model=' + caffemodel
	cmd += ' --model_def=' + deploy_prototxt
	cmd += ' --gpu --raw_scale=255'
	cmd += ' ' + image_file
  	# In detect.py, the .csv output
  	# code is buggy, and the hdf5 gave weird uint8 prediction values, so I
  	# pickled the pandas DataFrame instead.
	cmd += ' ' + output_filename
	system(cmd)
	return coordinates(output_filename)


#detect('_temp/det_input.txt' ,'_temp/_output.h5')

def coordinates(output_filename):

	df = pd.read_hdf(output_filename, 'df')
	with open('data/ilsvrc12/det_synset_words.txt') as f:
	    labels_df = pd.DataFrame([
	        {
	            'synset_id': l.strip().split(' ')[0],
	            'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
	        }
	        for l in f.readlines()
	    ])
	labels_df.sort('synset_id')
	predictions_df = pd.DataFrame(np.vstack(df.prediction.values), columns=labels_df['name'])
	k = 2
	max_s = predictions_df.max(0)
	max_s.sort(ascending=False)
	topk_objects = max_s[:k]
	
	coords = []
	for obj_index in range(k):
		obj = topk_objects.index[obj_index]
		print obj
		
		i = predictions_df[obj].argmax()
		
		det = df.iloc[i]
		coords1 = (det['xmin'], det['ymin']), det['xmax'] - det['xmin'], det['ymax'] - det['ymin']
		print coords1
		coords.append(coords1)
	return coords
