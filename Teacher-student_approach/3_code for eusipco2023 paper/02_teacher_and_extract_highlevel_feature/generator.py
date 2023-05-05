import numpy as np
import os
from itertools import islice
import sys
import re
from natsort import natsorted, ns
from hypara_model import *
import random
from PIL import Image
from numpy import asarray


class generator(object):

    def __init__(self, train_dir_01, test_dir_01, train_dir_02, test_dir_02, train_dir_03, test_dir_03):  

        self.train_dir_01      = train_dir_01
        self.test_dir_01       = test_dir_01

        self.train_dir_02      = train_dir_02
        self.test_dir_02       = test_dir_02

        self.train_dir_03      = train_dir_03
        self.test_dir_03       = test_dir_03

        self.class_num      = hypara().class_num
        self.label_dict     = hypara().label_dict

        self.train_file_id  = np.random.RandomState(seed=42).permutation(self.get_file_num(self.train_dir_01))
        self.test_file_id   = np.random.RandomState(seed=42).permutation(self.get_file_num(self.test_dir_01))

    def get_single_file (self, file_num, data_dir_01, data_dir_02, data_dir_03): 
        #print(data_dir)
        file_list     = self.get_file_list(data_dir_01)
        file_name     = file_list[file_num]
        file_open_01  = os.path.join(data_dir_01, file_name)
        file_open_02  = os.path.join(data_dir_02, file_name)
        file_open_03  = os.path.join(data_dir_03, file_name)

        o_data_01 = np.load(file_open_01)  
        o_data_02 = np.load(file_open_02)  
        o_data_03 = np.load(file_open_03)  

        o_data_01 = asarray(o_data_01)
        o_data_02 = asarray(o_data_02)
        o_data_03 = asarray(o_data_03)

        return o_data_01, o_data_02, o_data_03, file_name

    def get_batch (self, batch_num, batch_size, is_mixup, is_train):
        if is_train:
            file_list    = self.get_file_list(self.train_dir_01)
        else:
            file_list    = self.get_file_list(self.test_dir_01)
        #print(train_file_list)
        #exit()

        nImage = 0
        for ind in range(batch_num*batch_size, (batch_num+1)*batch_size):
            # open file
            if is_train:
                file_name = file_list[self.train_file_id[ind]]
                file_open_01 = os.path.join(self.train_dir_01, file_name)
                file_open_02 = os.path.join(self.train_dir_02, file_name)
                file_open_03 = os.path.join(self.train_dir_03, file_name)
            else:
                file_name = file_list[self.test_file_id[ind]]
                file_open_01 = os.path.join(self.test_dir_01, file_name)
                file_open_02 = os.path.join(self.test_dir_02, file_name)
                file_open_03 = os.path.join(self.test_dir_03, file_name)
            #print(file_open)
   
            #create label
            expectedClass = np.zeros([1,self.class_num])

            pattern = file_name.split('_')[-1]
            class_name = file_name.split('_'+pattern)[0]
            #print(class_name)

            nClass = self.label_dict[class_name]
            expectedClass[0,nClass] = 1
            #print(expectedClass, nClass)
            #exit()

            #create data
            one_image_01 = np.load(file_open_01)  
            one_image_02 = np.load(file_open_02)  
            one_image_03 = np.load(file_open_03)  

            one_image_01 = asarray(one_image_01)
            one_image_02 = asarray(one_image_02)
            one_image_03 = asarray(one_image_03)

            if (nImage == 0):
               seq_x_01 = one_image_01
               seq_x_02 = one_image_02
               seq_x_03 = one_image_03
               seq_y = expectedClass
            else:            
               seq_x_01 = np.concatenate((seq_x_01, one_image_01), axis=0)  
               seq_x_02 = np.concatenate((seq_x_02, one_image_02), axis=0)  
               seq_x_03 = np.concatenate((seq_x_03, one_image_03), axis=0)  
               seq_y = np.concatenate((seq_y, expectedClass), axis=0)  
            nImage += 1

        if is_mixup:
            o_data_01, o_data_02, o_data_03, o_label = self.mixup_aug(seq_x_01, seq_x_02, seq_x_03, seq_y, 0.4)
        else:

        #[nS, nW, nL, nC] = np.shape(o_data)
        #NewLength = nL - 10
        #for j in range(o_data.shape[0]):
        #    StartLoc1 = np.random.randint(0,o_data.shape[2]-NewLength)
        #    o_data[j, 0:NewLength, 0:NewLength, :] = o_data[j, StartLoc1:StartLoc1+NewLength, StartLoc1:StartLoc1+NewLength, :]
        #o_data = o_data[:, 0:NewLength, 0:NewLength, :]

            o_data_01  = seq_x_01
            o_data_02  = seq_x_02
            o_data_03  = seq_x_03
            o_label    = seq_y


        #print(np.shape(o_data), np.shape(o_label))
        #exit()
        
        return o_data_01, o_data_02, o_data_03, o_label

    def get_batch_num(self, batch_size, data_dir):
        return int(self.get_file_num(data_dir)/batch_size) #+ 1
    
    def get_file_num(self, data_dir):
        return len(os.listdir(data_dir))

    def get_file_list(self, data_dir):
        file_list = []
        org_file_list = os.listdir(data_dir)
        for i in range(0,len(org_file_list)):
           isHidden=re.match("\.",org_file_list[i])
           if (isHidden is None):
              file_list.append(org_file_list[i])
        #natsorted(file_list)
        file_list.sort()
        return file_list

    #---- online mixup data augmentation
    def mixup_aug(self, i_data_01, i_data_02, i_data_03, i_label, beta=0.4):
        half_batch_size = round(np.shape(i_data_01)[0]/2)
        #print(half_batch_size, np.shape(i_data))

        x01_1  = i_data_01[:half_batch_size,:]
        x01_2  = i_data_01[half_batch_size:,:]

        x02_1  = i_data_02[:half_batch_size,:]
        x02_2  = i_data_02[half_batch_size:,:]

        x03_1  = i_data_03[:half_batch_size,:]
        x03_2  = i_data_03[half_batch_size:,:]
        #print(np.shape(x1))
        #exit()


        #Translation 
        #[nS, nW, nL, nC] = np.shape(i_data)
        #NewLength = nL - 10
        #for j in range(x1.shape[0]):
        #    StartLoc1 = np.random.randint(0,x1.shape[2]-NewLength)
        #    StartLoc2 = np.random.randint(0,x2.shape[2]-NewLength)
        #    x1[j, 0:NewLength, 0:NewLength, :] = x1[j, StartLoc1:StartLoc1+NewLength, StartLoc1:StartLoc1+NewLength, :]
        #    x2[j, 0:NewLength, 0:NewLength, :] = x2[j, StartLoc2:StartLoc2+NewLength, StartLoc2:StartLoc2+NewLength, :]
        #x1 = x1[:, 0:NewLength, 0:NewLength, :]
        #x2 = x2[:, 0:NewLength, 0:NewLength, :]


        ##cut-off data augmentation
        #for j in range(x1.shape[0]):
        #    # spectrum augment
        #    for c in range(x1.shape[3]):
        #        x1[j, :, :, c] = self.frequency_masking(x1[j, :, :, c])
        #        x1[j, :, :, c] = self.time_masking(x1[j, :, :, c])
        #        x2[j, :, :, c] = self.frequency_masking(x2[j, :, :, c])
        #        x2[j, :, :, c] = self.time_masking(x2[j, :, :, c])

        ## random add noise
        #mean = 0
        #sigma = 2
        #for j in range(x1.shape[0]):
        #    nb,row,col,ch= x1.shape
        #    gauss_noise_1 = np.random.normal(mean,sigma,(row,col,ch))
        #    gauss_noise_1 = gauss_noise_1.reshape(1,row,col,ch)
        #    x1[j, :, :, :] = x1[j, :, :, :] +  gauss_noise_1

        #    nb,row,col,ch= x2.shape
        #    gauss_noise_2 = np.random.normal(mean,sigma,(row,col,ch))
        #    gauss_noise_2 = gauss_noise_2.reshape(1,row,col,ch)
        #    x2[j, :, :, :] = x2[j, :, :, :] +  gauss_noise_2


        y1  = i_label[:half_batch_size,:]
        y2  = i_label[half_batch_size:,:]

        # Beta dis
        #b   = np.random.beta(beta, beta, half_batch_size)
        b = np.random.uniform(0.2, 0.4, half_batch_size)
        X_b = b.reshape(half_batch_size, 1)
        y_b = b.reshape(half_batch_size, 1)

        xb01_mix   = x01_1*X_b     + x01_2*(1-X_b)
        xb01_mix_2 = x01_1*(1-X_b) + x01_2*X_b

        xb02_mix   = x02_1*X_b     + x02_2*(1-X_b)
        xb02_mix_2 = x02_1*(1-X_b) + x02_2*X_b

        xb03_mix   = x03_1*X_b     + x03_2*(1-X_b)
        xb03_mix_2 = x03_1*(1-X_b) + x03_2*X_b


        yb_mix   = y1*y_b     + y2*(1-y_b)
        yb_mix_2 = y1*(1-y_b) + y2*y_b

        ## Uniform dis
        ##l = np.random.randint(10, 35, size=half_batch_size)/100  #from 0.1 to 0.35
        #l   = np.random.random(half_batch_size)
        #X_l = l.reshape(half_batch_size, 1, 1, 1)
        #y_l = l.reshape(half_batch_size, 1)

        #xl_mix   = x1*X_l     + x2*(1-X_l)
        #xl_mix_2 = x1*(1-X_l) + x2*X_l

        #yl_mix   = y1* y_l    + y2 * (1-y_l)
        #yl_mix_2 = y1*(1-y_l) + y2*y_l

        #o_data     = np.concatenate((xb_mix,    x1,    xl_mix,    xb_mix_2,    x2,    xl_mix_2),    0)
        #o_label    = np.concatenate((yb_mix,    y1,    yl_mix,    yb_mix_2,    y2,    yl_mix_2),    0)
        o_data_01  = np.concatenate((xb01_mix,  x01_1,     xb01_mix_2,    x01_2),    0)
        o_data_02  = np.concatenate((xb02_mix,  x02_1,     xb02_mix_2,    x02_2),    0)
        o_data_03  = np.concatenate((xb03_mix,  x03_1,     xb03_mix_2,    x03_2),    0)
        o_label    = np.concatenate((yb_mix,    y1,        yb_mix_2,      y2),       0)

        return o_data_01, o_data_02, o_data_03, o_label

    def frequency_masking(self, mel_spectrogram, frequency_masking_para=20, frequency_mask_num=1):
        fbank_size = mel_spectrogram.shape

        for i in range(frequency_mask_num):
            f = random.randrange(0, frequency_masking_para)
            f0 = random.randrange(0, fbank_size[0] - f)

            if (f0 == f0 + f):
                continue

            mel_spectrogram[f0:(f0+f),:] = 0
        return mel_spectrogram


    def time_masking(self, mel_spectrogram, time_masking_para=20, time_mask_num=1):
        fbank_size = mel_spectrogram.shape

        for i in range(time_mask_num):
            t = random.randrange(0, time_masking_para)
            t0 = random.randrange(0, fbank_size[1] - t)

            if (t0 == t0 + t):
                continue

            mel_spectrogram[:, t0:(t0+t)] = 0
        return mel_spectrogram
        
