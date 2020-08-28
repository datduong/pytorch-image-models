

import os,sys,re,pickle
import numpy as np 
import pandas as pd 


def convert_score_01_range (x,a,b=0.5): 
  # Diagnosis confidences are expressed as floating-point values in the closed interval [0.0, 1.0], where 0.5 is used as the binary classification threshold. Note that arbitrary score ranges and thresholds can be converted to the range of 0.0 to 1.0, with a threshold of 0.5, trivially using the following sigmoid conversion:
  # 1 / (1 + e^(-(a(x - b))))
  # where x is the original score, b is the binary threshold, and a is a scaling parameter (i.e. the inverse measured standard deviation on a held-out dataset). Predicted responses should set the binary threshold b to a value where the classification system is expected to achieve 89% sensitivity, although this is not required.
  # 1 / (1 + e^(-(a(x - b))))
  return 1 / (1 + np.exp(-((x - b)/a)))


def convert_score_01_range_vertical (x): # ! do not use, looks wrong
  new_x = np.zeros(x.shape)
  for i in range(x.shape[1]) : # go over col
    a = np.std( x[:,i] )
    new = 1 / (1 + np.exp(-((x[:,i] - .5)/a)))
    new_x[:,i] = new
  return new_x


def convert_score_01_range_horizontal (x): 
  new_x = np.zeros(x.shape)
  for i in range(x.shape[0]) : # go over col
    a = np.std( x[i,:] )
    new = 1 / (1 + np.exp(-((x[i,:] - .5)/a)))
    new_x[i,:] = new
  return new_x


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
  prediction = reorder_col(prediction)
  num_sample = prediction.shape[0]
  # a_each_row = np.std(prediction,axis=1).reshape ( (num_sample,1)) # reshape to do broadcast
  # output = convert_score_01_range(prediction,a_each_row)
  output = convert_score_01_range_horizontal ( prediction )
  if average_augment : # ! take average over all augmentations 
    output = average_over_augmentation (output)
  #
  fout = open ( output_name , 'w' )
  if len(obs_name) > 0 :
    fout.write('image,MEL,NV,BCC,AKIEC,BKL,DF,VASC\n')
    for index,name in enumerate(obs_name): 
      fout.write(name+','+ ','.join ( str(x) for x in output[index] ) + '\n')
    # 
  else: 
    fout.write('MEL,NV,BCC,AKIEC,BKL,DF,VASC\n')
    for index in np.arange(output.shape[0]): 
      fout.write(','.join ( str(x) for x in output[index] ) + '\n')
    # 
  fout.close() 
  

