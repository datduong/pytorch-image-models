

import os,sys,re,pickle
import numpy as np 
import pandas as pd 
from copy import deepcopy

from sklearn.metrics import accuracy_score

def convert_score_01_range (x,a,b=0.5): 
  # Diagnosis confidences are expressed as floating-point values in the closed interval [0.0, 1.0], where 0.5 is used as the binary classification threshold. Note that arbitrary score ranges and thresholds can be converted to the range of 0.0 to 1.0, with a threshold of 0.5, trivially using the following sigmoid conversion:
  # 1 / (1 + e^(-(a(x - b))))
  # where x is the original score, b is the binary threshold, and a is a scaling parameter (i.e. the inverse measured standard deviation on a held-out dataset). Predicted responses should set the binary threshold b to a value where the classification system is expected to achieve 89% sensitivity, although this is not required.
  # 1 / (1 + e^(-(a(x - b))))
  return 1 / (1 + np.exp(-((x - b)/a)))


def convert_score_01_range_vertical (x): 
  new_x = np.zeros(x.shape)
  for i in range(x.shape[1]) : # go over col
    a = np.std( x[:,i] )
    new = 1 / (1 + np.exp(-((x[:,i] - .5)/a)))
    new_x[:,i] = new
  return new_x


def convert_score_01_range_horizontal (x): # ! do not use, looks wrong
  new_x = np.zeros(x.shape)
  for i in range(x.shape[0]) : # go over col
    a = np.std( x[i,:] )
    new = 1 / (1 + np.exp(-((x[i,:] - .5)/a)))
    new_x[i,:] = new
  return new_x


def convert_max_1_other_0 (a): 
  b = np.zeros_like(a)
  b[np.arange(len(a)), a.argmax(1)] = 1
  b = b + .0001
  b[b > 1] = .9999 # site won't take integer ? 
  return b


def find_best_convert (x,true_label): 
  best_b = 0
  new_x = np.zeros(x.shape)
  best_score = 0
  a = np.std(np.squeeze(np.asarray(x)))
  for b in np.arange(-10,10,.1): 
    new = 1 / (1 + np.exp(-((x - b)/a)))
    current_score = accuracy_score(true_label, np.round(new))
    if current_score > best_score: 
      new_x = deepcopy(new)
      best_b = b
  return new_x, best_b


def find_best_convert_each_i (x,true_label): 
  best_b = np.zeros(x.shape[1])
  new_x = np.zeros(x.shape)
  for i in range(x.shape[1]) : # go over each col
    a = np.std( x[:,i] ) ## go down col, take std
    best_score = 0
    for b in np.arange(-1,1,.1): 
      new = 1 / (1 + np.exp(-((x[:,i] - b)/a)))
      current_score = accuracy_score(true_label[:,i], np.round(new))
      if current_score > best_score: 
        new_x[:,i] = new
        best_b[i] = b
  return new_x, best_b

  
def reorder_col (inputs,pytorch_label=['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']): 
  # File columns must be:
  # image: an input image identifier of the form ISIC_
  # MEL: “Melanoma” diagnosis confidence
  # NV: “Melanocytic nevus” diagnosis confidence
  # BCC: “Basal cell carcinoma” diagnosis confidence
  # AKIEC: “Actinic keratosis / Bowen’s disease (intraepithelial carcinoma)” diagnosis confidence
  # BKL: “Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis)” diagnosis confidence
  # DF: “Dermatofibroma” diagnosis confidence
  # VASC: “Vascular lesion” diagnosis confidence

  # ! pytorch sort label by alphabet, so we need to reorder 
  gold_label_order = ['mel','nv','bcc','akiec','bkl','df','vasc']
  new_order = [ pytorch_label.index(i) for i in gold_label_order ]
  return inputs[ : , new_order ] # swap col


def average_over_augmentation (output,default_aug_num=10) : 
  new_output = np.zeros( ( output.shape[0]//default_aug_num , 7 ) ) # must be divisible by @default_aug_num
  for i in range(new_output.shape[0]): 
    new_output[i] = np.mean( output[ i*10 : (i+1)*10 ] , axis=0 )
  return new_output


def save_output_csv(prediction, obs_name, output_name, average_augment=False): 
  output = reorder_col(prediction) # @prediction should be 0-1 range
  num_sample = output.shape[0]

  if average_augment : # ! take average over all augmentations in off-line mode 
    output = average_over_augmentation (output)
  
  fout = open ( output_name , 'w' )
  if len(obs_name) > 0 :
    fout.write('image,MEL,NV,BCC,AKIEC,BKL,DF,VASC\n')
    for index,name in enumerate(obs_name): 
      fout.write(name+','+ ','.join ( str(x) for x in output[index] ) + '\n')
  else: 
    fout.write('MEL,NV,BCC,AKIEC,BKL,DF,VASC\n')
    for index in np.arange(output.shape[0]): 
      fout.write(','.join ( str(x) for x in output[index] ) + '\n')
  
  fout.close() 
  

def softmax(X, theta = 1.0, axis = 1): # https://nolanbconaway.github.io/blog/2017/softmax-numpy.html
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p