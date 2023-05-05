#---- general packages
import numpy as np
import os
import argparse
import math
import scipy.io
import re
import time
import datetime
import sys
from sklearn import datasets, svm, metrics
#----- layers
import tensorflow as tf
from tensorflow.keras.applications import DenseNet201, Xception, NASNetLarge, NASNetLarge, VGG19, InceptionV3, MobileNetV2, ResNet50V2, MobileNet, VGG16, EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras import initializers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.layers import Add, Average
from tensorflow.keras.layers import Concatenate

from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Reshape, multiply, Permute, Lambda
from tensorflow.keras.activations import sigmoid

#----- generator
#sys.path.append('./02_models/')
#from test_model import *
from generator import *
from hypara_model import *

#----- main
def trippleAttention(x): # 8x8
    ## TA - Tripple Attention
    nHead = 32
    nKeyDim = 16
    print(np.shape(x))

    [nS, nF, nT, nC] = np.shape(x)

    # channel
    tl1 = tf.math.reduce_mean(x, axis=-1) + tf.math.reduce_max(x, axis=-1)# 8x8
    tl1 = MultiHeadAttention(num_heads=nHead, key_dim=nKeyDim)(tl1, tl1)           # 8x8
    tl1 = sigmoid(tl1)           # 8x8
    tl1 = Reshape((nF,nT,1))(tl1)  # 8x8x1
    tl1 = x * tl1                # 8x8xc * 8x8x1 -> 8x8xc

      # width
    tl2 = tf.math.reduce_mean(x, axis=-2) + tf.math.reduce_max(x, axis=-2)# 8xc
    tl2 = MultiHeadAttention(num_heads=nHead, key_dim=nKeyDim)(tl2, tl2)          # 8xc
    tl2 = sigmoid(tl2)            # 8xcx1
    tl2 = Reshape((nF,1,nC))(tl2)   # 8x1xc
    tl2 = x * tl2                 # 8x8xc * 8x1xc -> 8x8xc

      # height
    tl3 = tf.math.reduce_mean(x, axis=-3) + tf.math.reduce_max(x, axis=-3)# 8xc
    tl3 = MultiHeadAttention(num_heads=nHead, key_dim=nKeyDim)(tl3, tl3)          # 8xc
    tl3 = sigmoid(tl3)            # 8xcx1
    tl3 = Reshape((1,nT,nC))(tl3)   # 8x1xc
    tl3 = x * tl3                 # 8x8xc * 1x8xc -> 8x8xc
      # average
    t = Average()([tl1, tl2, tl3]) # 8x8xc
    t = GlobalAveragePooling2D()(t) #channel
    return t # c

def main():
    print("\n ==================================================================== SETUP PARAMETERS...")
    print("-------- PARSER:")
    parser = init_argparse()
    args   = parser.parse_args()
    OUT_DIR          = args.out_dir
    IS_TRAINING      = args.is_training
    IS_TESTING       = args.is_testing
    IS_EXTRACT       = args.is_extract

    print("-------- Hyper parameters:")
    #nF               = hypara().nF
    #nC               = hypara().nC
    #nT               = hypara().nT
    BATCH_SIZE       = hypara().batch_size
    NUM_EPOCHS       = hypara().epoch_num
    N_CLASS          = hypara().class_num
    LEARNING_RATE    = hypara().learning_rate
    CHECKPOINT_EVERY = hypara().check_every
    IS_MIXUP         = hypara().is_aug
    START_BATCH      = hypara().start_batch
    
    #Setting directory
    print("\n =============== Directory Setting...")
    stored_dir = os.path.abspath(os.path.join(os.path.curdir, OUT_DIR))
    print("+ Writing to {}\n".format(stored_dir))

    best_model_dir = os.path.join(stored_dir, "model")
    print("+ Best model Dir: {}\n".format(best_model_dir))
    if not os.path.exists(best_model_dir):
        os.makedirs(best_model_dir)
    best_model_file = os.path.join(best_model_dir, "best_model.h5")

    #Random seed
    tf.random.set_seed(0) #V2
    
    #Instance model or reload an available model
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():

        if os.path.isfile(best_model_file):
            with open(os.path.join(stored_dir,"train_acc_log.txt"), "a") as text_file:
                text_file.write("---------- Latest Model Loaded from dir: {}\n".format(best_model_dir))
            print("\n=============== Latest Model Loaded from dir: {}" .format(best_model_dir))
            model = tf.keras.models.load_model(best_model_file)
            #for layer in model.layers:
            #    if re.search('block[0-9]',layer.name):
            #        layer.trainable = False
            #        print(layer.name)
        else:
            print("\n=============== New model instance is created \n")
            #base_model = DenseNet201(weights=None, include_top=False)
            #base_model = DenseNet201(weights='imagenet', include_top=False)
            #base_model = InceptionV3(weights='imagenet', include_top=False)
            base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(246,246,3))
            #for layer in base_model.layers: 
            #    if hasattr(layer, 'kernel_initializer'): 
            #        layer.kernel_initializer=initializers.TruncatedNormal(mean=0.0,stddev=0.1)
            #    if hasattr(layer, 'kernel_regularizer'):
            #        layer.kernel_regularizer=regularizers.l2(1e-5)
            #    if hasattr(layer, 'bias_initializer'):
            #        layer.bias_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1)
            #    if hasattr(layer, 'bias_regularizer'):
            #        layer.bias_regularizer=regularizers.l2(1e-5)
            
            #x = base_model.output
            #block5_x = base_model.get_layer('block5c_add').output  
            #block6_x = base_model.get_layer('block6d_add').output  
            block7_x = base_model.output                           

            #block5_x = Conv2D(filters=1280, kernel_size=2, strides=2)(block5_x) 
            #block6_x = Conv2D(filters=1280, kernel_size=1, strides=1)(block6_x) 

            #print('# block 5: ', np.shape(block5_x)) # 1024
            #print('# block 6: ', np.shape(block6_x)) # 1920
            #print('# block 7: ', np.shape(block7_x)) # 1920

            #block5_x = trippleAttention(block5_x)
            #block6_x = trippleAttention(block6_x)
            x = trippleAttention(block7_x)

            #print('# block 5: ', np.shape(block5_x)) # 1024
            #print('# block 6: ', np.shape(block6_x)) # 1920
            #print('# block 7: ', np.shape(block7_x)) # 1920

            #x = Concatenate(axis=-1)([block5_x, block6_x, block7_x]) 
            #x = block5_x + block6_x + block7_x
            x = Dropout(0.2)(x)

            x = Dense(512, 
                      activation='relu',
                      kernel_initializer=initializers.TruncatedNormal(mean=0.0,stddev=0.1),
                      kernel_regularizer=regularizers.l2(1e-5),
                      bias_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                      bias_regularizer=regularizers.l2(1e-5)
                     )(x)
            x = Dropout(0.2)(x)

            predictions = Dense(45, 
                                activation='softmax',
                                kernel_initializer=initializers.TruncatedNormal(mean=0.0,stddev=0.1),
                                kernel_regularizer=regularizers.l2(1e-5),
                                bias_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                                bias_regularizer=regularizers.l2(1e-5)
                               )(x)

            model = Model(inputs=base_model.input, outputs=predictions)
            #for layer in base_model.layers:
            #    layer.trainable = False
        model.summary()
        exit()

        # initialize the optimizer and compile the model
        opt = tf.keras.optimizers.Adam(LEARNING_RATE)
        #model.compile(loss="kullback_leibler_divergence", optimizer=opt, metrics=["accuracy"])   
        model.compile(loss=tf.keras.losses.kullback_leibler_divergence, optimizer=opt, metrics=["accuracy"])   

    old_file_valid_acc  = 0
    old_eva_acc_avg = 0

    test_threshold   = 0.9
    generator_ins    = generator(args.train_dir, args.test_dir)
    n_valid          = generator_ins.get_file_num(args.test_dir)
    batch_num        = generator_ins.get_batch_num(BATCH_SIZE, args.train_dir)
    batch_test       = 100
    batch_num_test   = generator_ins.get_batch_num(batch_test, args.test_dir)
    #print(batch_num)
    #exit()

    if IS_TRAINING == 'yes':
        for nEpoch in range(NUM_EPOCHS):
            print("\n=======================  Epoch is", nEpoch, ";============================")
            for nBatchTrain in range(START_BATCH, batch_num):
                x_train_batch, y_train_batch, n_image = generator_ins.get_batch(nBatchTrain, BATCH_SIZE, IS_MIXUP, True) #    
                #print(np.shape(x_train_batch))
                #exit()
                [train_loss, train_acc] = model.train_on_batch(x_train_batch, y_train_batch,reset_metrics=True)
                if (nBatchTrain % CHECKPOINT_EVERY == 0):  
                    print("Epoch: {}, TRAIN Accuracy:{}".format(nEpoch,train_acc))
                    with open(os.path.join(stored_dir,"train_acc_log.txt"), "a") as text_file:
                        text_file.write("Epoch:{}, TRAIN ACC:{} \n".format(nEpoch, train_acc))

                    if(train_acc >= test_threshold):   
                    #    eva_acc_avg = 0
                    #    for nBatchEva in range(0,START_BATCH):
                    #        x_eva_batch, y_eva_batch, n_image = generator_ins.get_batch(nBatchEva, BATCH_SIZE, False) #NO MIXUP FOR EVA    
                    #        [eva_loss, eva_acc] = model.evaluate(x_eva_batch, y_eva_batch, verbose=0)
                    #        eva_acc_avg += eva_acc
                    #    eva_acc_avg = eva_acc_avg/START_BATCH
                    #    #print("Epoch: {}, EVA Accuracy:{}".format(nEpoch,eva_acc_avg))

                    #    #01/ Save model when success evaluating on batches
                    #    if (eva_acc_avg > old_eva_acc_avg):
                    #        old_eva_acc_avg  = eva_acc_avg 

                    #        model.save(best_model_file)
                    #        with open(os.path.join(stored_dir,"train_acc_log.txt"), "a") as text_file:
                    #            text_file.write("Save best model at Epoch: {}; Score {}\n".format(nEpoch, old_eva_acc_avg))

                        if IS_TESTING == 'yes':  
                            eva_acc_avg = 0
                            for nBatchEva in range(0,batch_num_test):
                                x_eva_batch, y_eva_batch, n_image = generator_ins.get_batch(nBatchEva, batch_test, False, False) #NO MIXUP FOR EVA    
                                [eva_loss, eva_acc] = model.evaluate(x_eva_batch, y_eva_batch, verbose=0)
                                eva_acc_avg += eva_acc
                            eva_acc_avg = eva_acc_avg/batch_num_test

                            #01/ Save model when success evaluating on batches
                            if (eva_acc_avg > old_eva_acc_avg):
                                old_eva_acc_avg  = eva_acc_avg 
                                model.save(best_model_file)
                                with open(os.path.join(stored_dir,"train_acc_log.txt"), "a") as text_file:
                                    text_file.write("Save best model at Epoch: {}; Score {}\n".format(nEpoch, old_eva_acc_avg))

                                #if old_eva_acc_avg > 0.94:    
                                #    if (IS_EXTRACT == 'yes'):
                                #        with open(os.path.join(stored_dir,"train_acc_log.txt"), "a") as text_file:
                                #            text_file.write("Extraction ....\n")

                                #        final_layer_stored_matrix = np.zeros([n_valid, N_CLASS]) #file_num x nClass
                                #        final_layer_res_dir = os.path.abspath(os.path.join(stored_dir, "01_pre_res"))
                                #        if not os.path.exists(final_layer_res_dir):
                                #            os.makedirs(final_layer_res_dir)
                                #        final_layer_file = os.path.abspath(os.path.join(final_layer_res_dir, "pre_train_res"))
                                #        file_name_list = []       
                                #        for nFileValid in range(0,n_valid):
                                #            x_valid_batch, valid_file_name = generator_ins.get_single_file(nFileValid, args.test_dir)
                                #    
                                #            file_name_list.append(valid_file_name)
                                #            x_selected = x_valid_batch#.astype(np.float16)
                                #            valid_end_output = model.predict(x_selected)
                                #            sum_valid_end_output = np.sum(valid_end_output, axis=0) #1xnClass
                                #            final_layer_stored_matrix[nFileValid,:] = sum_valid_end_output
                                #        np.savez(final_layer_file, matrix=final_layer_stored_matrix, file_name=file_name_list)

                                        #exit() # exit after extracting

                        #---------------------------------------- Test on each file
                        #if IS_TESTING == 'yes':  
                        #    file_valid_acc   = 0
                        #    fuse_matrix      = np.zeros([N_CLASS, N_CLASS])
                        #    valid_metric_reg = np.zeros(n_valid)
                        #    valid_metric_exp = np.zeros(n_valid)
                        # 
                        #    for nFileValid in range(0,n_valid):
                        #    #for nFileValid in range(0,10):
                        #        x_valid_batch, valid_file_name = generator_ins.get_single_file(nFileValid, args.test_dir)
                        #        #print(np.shape(x_valid_batch))
                        #        #exit()
                        #        #expected  
                        #            
                        #        pattern = valid_file_name.split('_')[-1]
                        #        class_name = valid_file_name.split('_'+pattern)[0]

                        #        #class_name = valid_file_name.split('_')[-2]
                        #        valid_res_exp  = hypara().label_dict[class_name]
                        #        #recognized  
                        #        #print(type(x_valid_batch))
                        #        #x_valid_batch = x_valid_batch[0:2,:,:,:].astype(np.float16)   #0:2 --> only check 0:2
                        #        #x_valid_batch = x_valid_batch.astype(np.float16)   
                        #        valid_end_output = model.predict(x_valid_batch)
                        #
                        #        #print(np.shape(valid_end_output))
                        #        #exit()

                        #        sum_valid_end_output = np.sum(valid_end_output, axis=0) #1xnClass
                        #        valid_res_reg        = np.argmax(sum_valid_end_output)
                        #        
                        #        #Compute acc
                        #        valid_metric_reg[nFileValid] = int(valid_res_reg)
                        #        valid_metric_exp[nFileValid] = int(valid_res_exp)
                        # 
                        #        #For general report
                        #        fuse_matrix[valid_res_exp, valid_res_reg] = fuse_matrix[valid_res_exp, valid_res_reg] + 1
                        #        if(valid_res_reg == valid_res_exp):
                        #            file_valid_acc = file_valid_acc + 1
 
                        #    # For general report
                        #    file_valid_acc  = file_valid_acc*100/n_valid

                        #    #for sklearn metric
                        #    print("Classification report for classifier \n%s\n"
                        #          % (metrics.classification_report(valid_metric_exp, valid_metric_reg)))
                        #    cm = metrics.confusion_matrix(valid_metric_exp, valid_metric_reg)
                        #    print("Confusion matrix:\n%s" % cm)
                        # 
                        #    with open(os.path.join(stored_dir,"valid_acc_log.txt"), "a") as text_file:
                        #        text_file.write("========================== VALIDATING ONLY =========================================== \n\n")
                        #        text_file.write("On File Final Accuracy:  {}%\n".format(file_valid_acc))
                        #        text_file.write("{0} \n".format(fuse_matrix))
                        #        text_file.write("========================================================================== \n\n")

                        #    if file_valid_acc > old_file_valid_acc:
                        #        old_file_valid_acc = file_valid_acc
                        #        model.save(best_model_file)


    #--------- Testing only
    if IS_TESTING == 'yes':  
        eva_acc_avg = 0
        for nBatchEva in range(0,batch_num_test):
            x_eva_batch, y_eva_batch, n_image = generator_ins.get_batch(nBatchEva, batch_test, False, False) #NO MIXUP FOR EVA    
            [eva_loss, eva_acc] = model.evaluate(x_eva_batch, y_eva_batch, verbose=0)
            eva_acc_avg += eva_acc
        eva_acc_avg = eva_acc_avg/batch_num_test
        with open(os.path.join(stored_dir,"train_acc_log.txt"), "a") as text_file:
            text_file.write("------------------------- Testing Score {}\n".format(eva_acc_avg))

    #--------- Extract embbeding
    if (IS_EXTRACT == 'yes'):
        final_layer_stored_matrix = np.zeros([n_valid, N_CLASS]) #file_num x nClass
    
        final_layer_res_dir = os.path.abspath(os.path.join(stored_dir, "01_pre_res"))
        if not os.path.exists(final_layer_res_dir):
            os.makedirs(final_layer_res_dir)
        final_layer_file = os.path.abspath(os.path.join(final_layer_res_dir, "pre_train_res"))
        file_name_list = []       
        for nFileValid in range(0,n_valid):
            x_valid_batch, valid_file_name = generator_ins.get_single_file(nFileValid, args.test_dir)
    
            file_name_list.append(valid_file_name)
            x_selected = x_valid_batch.astype(np.float16)
            valid_end_output = model.predict(x_selected)
            sum_valid_end_output = np.sum(valid_end_output, axis=0) #1xnClass
            final_layer_stored_matrix[nFileValid,:] = sum_valid_end_output
        np.savez(final_layer_file, matrix=final_layer_stored_matrix, file_name=file_name_list)
def init_argparse():
    parser = argparse.ArgumentParser(
        usage="%(prog)s --out_dir XXX --dev_dir XXX --train_dir XXX --eva_dir XXX --test_dir XXX",
        description="Set directory of spectrogram of dev/train/eva/test sets" 
    )
    parser.add_argument(
        "--out_dir", required=True,
        help=' --outdir <output directory>'
    )
    parser.add_argument(
        "--train_dir", required=True,
        help=' --train_dir <train spectrogram directory>'
    )
    parser.add_argument(
        "--test_dir", required=True,
        help=' --test_dir <test spectrogram directory>'
    )
    parser.add_argument(
        "--is_training", required=True,
        help=' --is_training <yes/no>'
    )
    parser.add_argument(
        "--is_testing", required=True,
        help=' --is_testing <yes/no>'
    )
    parser.add_argument(
        "--is_extract", required=True,
        help=' --is_extract <yes/no>'
    )
    return parser


if __name__ == "__main__":
    main()
