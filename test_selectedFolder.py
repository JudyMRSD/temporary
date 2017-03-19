import os
import sys

def add_path(path):
        if path not in sys.path:
                sys.path.insert(0,path)
caffe_path = '/home/harp/SaveData/Sharon/caffe/python'
#caffe_path = '/home/harp/Sharon/caffe/python'
add_path(caffe_path)

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#caffe_root = os.environ['CAFFE_ROOT']
#sys.path.insert(0,os.path.join(caffe_root, 'python'))
import caffe
import colorsys
from matplotlib.colors import LinearSegmentedColormap
import scipy.io as sio
import glob


# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
# im = Image.open('pascal/VOC2010/JPEGImages/2007_000129.jpg')
# im = Image.open('data/pascal/cat.jpg')     
CAFFEMODEL = sys.argv[1]
# load net
net = caffe.Net('deploy.prototxt', CAFFEMODEL, caffe.TEST)
# shape for input (data blob is N x C x H x W), set data

folder = sys.argv[2]
files = glob.glob(os.path.join(folder, '*.jpg'))
size = 640,360 
names = dict()
all_labels = ['background']+open('../../../data/APC40/classes_APC.txt').readlines()
outdir = sys.argv[3]

test_file_directory = open('../../../data/APC40/ImageSets/Segmentation/test.txt', 'r')
filelist = test_file_directory.readlines()
prefix = '/home/harp/SaveData/Sharon/data/APC40/JPEGImages/'
 
#A = open('../../../data/APC40/ImageSets/Segmentation/test.txt').read()

#for f in files:
#  f_str = str(f)
#  f_str = f_str[f_str.index("Images/") + len("Images/"):]
 
for f_str in filelist:
  f_str = f_str.strip()
  f = prefix + f_str + ".jpg"
  print f
  im = Image.open(f)
  im.thumbnail(size)
  in_ = np.array(im, dtype=np.float32)
  in_ = in_[:,:,::-1]
  in_ -= np.array((104.00698793,116.66876762,122.67891434))
  in_ = in_.transpose((2,0,1))

  net.blobs['data'].reshape(1, *in_.shape)
  net.blobs['data'].data[...] = in_
  # run net and take argmax for prediction
  net.forward()
  # import pdb; pdb.set_trace()
  out = net.blobs['score'].data[0].argmax(axis=0)
  scores=np.unique(out)
  labels=[all_labels[s] for s in scores]
  num_scores = len(scores)
  num_labels = len(all_labels)
  print "total labels = "+str(num_scores)
  print labels

  # visualization
  def rescore(c):
    """ rescore values from original score values (0-59) to values ranging from
    0 to num_scores-1 """
    return np.where(scores == c)[0][0]

  #rescore = np.vectorize(rescore)
  #painted = rescore(out)
  #painted  = out

  plt.rcParams['image.interpolation'] = 'none'  # don't interpolate
  plt.figure(figsize=(10, 10))
  plt.imshow(out,vmin =0, vmax = len(all_labels), cmap='hsv',interpolation='none')

  formatter = plt.FuncFormatter(lambda val, loc: all_labels[val])
  plt.colorbar(ticks=range(0, len(all_labels)), format=formatter)
  plt.clim(-0.5, len(all_labels) - 0.5)
  plt.imsave(outdir+os.path.basename(f), np.uint8(out), vmin=0, vmax = num_labels, cmap='hsv')

  sio.savemat(outdir+os.path.basename(f)+'.mat', mdict={'cdata': np.uint8(out)})
  print "Done! Output saved to "+sys.argv[3]+os.path.basename(f)



