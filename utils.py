import numpy as np
import nibabel as nib
import os
from glob import glob

################################################
# load image
################################################

def img_load(load_data_flag, img_suffix, dir_class1, dir_class2, dir_all, idno, train_ind, train_train_ind, train_val_ind, test_ind):

    if load_data_flag:

        # loading all the image together does not work for large dataset (due to the limit in memory)

        img_train_class1 = np.array(
            [nib.load(dir_class1+str(id)+img_suffix).get_data() for id in
             [idno[i] for i in train_ind[train_train_ind]]])
        img_train_class2 = np.array(
            [nib.load(dir_class2+str(id)+img_suffix).get_data() for id in
             [idno[i] for i in train_ind[train_train_ind]]])

        img_val_class1 = np.array(
            [nib.load(dir_class1+str(id)+img_suffix).get_data() for id in
             [idno[i] for i in train_ind[train_val_ind]]])
        img_val_class2 = np.array(
            [nib.load(dir_class2+str(id)+img_suffix).get_data() for id in
             [idno[i] for i in train_ind[train_val_ind]]])

        img_test_class1 = np.array(
            [nib.load(dir_class1 + str(id)+img_suffix).get_data() for id in
            [idno[i] for i in test_ind]])
        img_test_class2 = np.array(
            [nib.load(dir_class2 + str(id)+img_suffix).get_data() for id in
            [idno[i] for i in test_ind]])

        img_train = \
        np.zeros((img_train_class1.shape[0]*2,img_train_class1.shape[1],img_train_class1.shape[2],img_train_class1.shape[3]))
        img_train[range(0,img_train.shape[0],2),:,:,:]=img_train_class1
        img_train[(range(1,img_train.shape[0],2)),:,:,:]=img_train_class2
        img_val  = np.concatenate((img_val_class1,img_val_class2),axis=0)      
        img_test = np.concatenate((img_test_class1,img_test_class2),axis=0)

        img_train = np.expand_dims(img_train,axis=1)
        img_val   = np.expand_dims(img_val, axis=1)
        img_test  = np.expand_dims(img_test,axis=1)

        label_train = np.zeros(img_train.shape[0])
        label_train[range(0,img_train.shape[0],2)]=0
        label_train[range(1,img_train.shape[0],2)]=1        
        label_val  = np.concatenate((np.zeros(img_val.shape[0]/2),np.ones(img_val.shape[0]/2)),axis=0)
        label_test = np.concatenate((np.zeros(img_test.shape[0]/2),np.ones(img_test.shape[0]/2)),axis=0)

        np.save(dir_all+'img_train.npy',img_train)
        np.save(dir_all+'img_val.npy', img_val)
        np.save(dir_all+'img_test.npy',img_test)
        np.save(dir_all+'label_train.npy',label_train)
        np.save(dir_all+'label_val.npy', label_val)
        np.save(dir_all+'label_test.npy',label_test)
	
    else:
        img_train   = np.load(dir_all + 'img_train.npy')
        img_val     = np.load(dir_all+'img_val.npy')
        img_test    = np.load(dir_all+'img_test.npy')
        label_train = np.load(dir_all + 'label_train.npy')
        label_val   = np.load(dir_all+'label_val.npy')
        label_test  = np.load(dir_all+'label_test.npy')
       
    
    return img_train, img_val, img_test, label_train, label_val, label_test
 
################################################
# image normalization
################################################

def img_norm(img, int_max, int_min, int_mean):
    
    img[img < int_min]    = int_min
    img[img > int_max]    = int_max
    img = (img - int_mean)*1.0/(int_max - int_min)
    
    return img
