import numpy as np
import nibabel as nib
import os
from glob import glob
from sklearn.model_selection import StratifiedKFold
from build_network import *
from utils import *

####################################################################
# Predefined parameters
####################################################################

img_size       = [128, 128, 128] # size of ROI
int_min        = -1000  # min value for intensity normalization
int_max        = 0      # max value for intensity normalization
int_mean       = -800   # mean value for intensity normalization
load_data_flag = True

####################################################################
# Predefined directories
####################################################################

dir_class1  = '../sample_class1/'
dir_class2  = '../sample_class2/'
dir_all     = '../sample_all/'

img_suffix  = '.nii.gz' # suffix for 3D data
idno_list   = sorted(glob(dir_class1+'*'+img_suffix))
idno        = []

for id in range(0,len(idno_list)):
    idno.append(idno_list[id].split('/')[-1][:-len(img_suffix)])

####################################################################
# Split training, validation and test
####################################################################

# train vs. val vs. test = 4:1:1
skf_train_test = StratifiedKFold(n_splits=6,random_state=1)
skf_train_val  = StratifiedKFold(n_splits=5,random_state=1)

# change stratify_group if you have prior knowledge (e.g. demographic info) for stratified sampling
stratify_group = np.ones(len(idno))

####################################################################
# Begin training and evaluation
####################################################################

for train_ind,test_ind in skf_train_test.split(idno,stratify_group):
    
    # just access the first fold without a for loop
    train_train_ind, train_val_ind = list(skf_train_val.split([idno[i] for i in train_ind], stratify_group[train_ind]))[0]
      
    img_train, img_val, img_test, label_train, label_val, label_test = \
    img_load(load_data_flag, img_suffix, dir_class1, dir_class2, dir_all, idno, train_ind, train_train_ind, train_val_ind, test_ind)
    
    img_train = img_norm(img_train, int_max, int_min, int_mean)
    img_val   = img_norm(img_val, int_max, int_min, int_mean)
    img_test  = img_norm(img_test, int_max, int_min, int_mean)
        
    depth_level  = 4 # number of conv layers = depth_level * 2
    model        = train_network(img_train,img_val,label_train,label_val,img_size,depth_level)
    test_predict = model.predict(img_test)
    model.evaluate(img_test,label_test)
