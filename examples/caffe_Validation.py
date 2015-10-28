import numpy as np

import sys
import os

#sys.stdout =  open(os.devnull, "w")

import matplotlib
matplotlib.use('Agg')

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
import caffe.io
#plt.rcParams['figure.figsize'] = (10, 10)
#plt.rcParams['image.interpolation'] = 'nearest'
#plt.rcParams['image.cmap'] = 'gray'

# GPU mode
caffe.set_device(0)
caffe.set_mode_gpu()

net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

lines = [line.rstrip('\n') for line in open('val1.txt')]
print len(lines)

# set net to batch size of 50
net.blobs['data'].reshape(1,3,227,227)

for x in range(0, 1):
    net.blobs['data'].data[x] = transformer.preprocess('data', caffe.io.load_image(caffe_root + 'path/to/imagenet/val/' + lines[x]))

#net.blobs['data'].data[1] = transformer.preprocess('data', caffe.io.load_image(caffe_root + 'path/to/imagenet/val/ILSVRC2012_val_00001113.JPEG'))
#net.blobs['data'].data[2] = transformer.preprocess('data', caffe.io.load_image(caffe_root + 'path/to/imagenet/val/ILSVRC2012_val_00001114.JPEG'))
#net.blobs['data'].data[3] = transformer.preprocess('data', caffe.io.load_image(caffe_root + 'path/to/imagenet/val/ILSVRC2012_val_00001116.JPEG'))
#net.blobs['data'].data[4] = transformer.preprocess('data', caffe.io.load_image(caffe_root + 'path/to/imagenet/val/ILSVRC2012_val_00001118.JPEG'))
#import timeit

from datetime import datetime
startTime = datetime.now()
out = net.forward()
endTime = datetime.now()
#timeit net.forward()
#print("Predicted class is #{}.".format(out['prob'].argmax()))

# load labels
imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
try:
    labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
except:
    #../data/ilsvrc12/get_ilsvrc_aux.sh
    labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

# sort top k predictions from softmax output
top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
print net.blobs['prob'].shape
print blobproto_to_array(net.blobs['prob'])
#print labels[top_k]
top_j = net.blobs['prob'].data[1].flatten().argsort()[-1:-6:-1]
print labels[top_j]

#sys.stdout = sys.__stdout__

print endTime - startTime 

#timeit.repeat("net.forward()","from __main__ import net.forward, number=

#for k, v in net.params.items():
#	print k, v[0].data.shape 

