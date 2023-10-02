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
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras import initializers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling1D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate
#from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.layers import Add
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
            base_model = DenseNet201(weights='imagenet', include_top=False)
            #for layer in base_model.layers:
            #    if hasattr(layer, 'kernel_initializer'):
            #        layer.kernel_initializer=initializers.TruncatedNormal(mean=0.0,stddev=0.1)
            #    if hasattr(layer, 'kernel_regularizer'):
            #        layer.kernel_regularizer=regularizers.l2(1e-5)
            #    if hasattr(layer, 'bias_initializer'):
            #        layer.bias_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1)
            #    if hasattr(layer, 'bias_regularizer'):
            #        layer.bias_regularizer=regularizers.l2(1e-5)

            x = base_model.output
            x = GlobalAveragePooling2D()(x)
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
        #exit()

        # initialize the optimizer and compile the model
        opt = tf.keras.optimizers.Adam(LEARNING_RATE)
        #model.compile(loss="kullback_leibler_divergence", optimizer=opt, metrics=["accuracy"])   
        model.compile(loss=tf.keras.losses.kullback_leibler_divergence, optimizer=opt, metrics=["accuracy"])   

    old_file_valid_acc  = 0
    old_eva_acc_avg = 0

    test_threshold   = 0.9
    generator_ins    = generator(args.train_dir, args.test_dir)
    n_train          = generator_ins.get_file_num(args.train_dir)
    batch_num        = generator_ins.get_batch_num(BATCH_SIZE, args.train_dir)

    batch_test       = 100
    batch_num_test   = generator_ins.get_batch_num(batch_test, args.test_dir)

    n_valid          = generator_ins.get_file_num(args.test_dir)
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

                                if  nEpoch == NUM_EPOCHS-1:
                                    #------------------------------ EXTRACTION
                                    if (IS_EXTRACT == 'yes'):
                                        #---------------------------------------- Extract Prob
                                        final_layer_stored_matrix = np.zeros([n_valid, N_CLASS]) #file_num x nClass
                                    
                                        final_layer_res_dir = os.path.abspath(os.path.join(stored_dir, "01_pre_res"))
                                        if not os.path.exists(final_layer_res_dir):
                                            os.makedirs(final_layer_res_dir)
                                        final_layer_file = os.path.abspath(os.path.join(final_layer_res_dir, "pre_train_res"))
                                        file_name_list = []       
                                        for nFileValid in range(0,n_valid):
                                            x_valid_batch, valid_file_name = generator_ins.get_single_file(nFileValid, args.test_dir)
                                    
                                            file_name_list.append(valid_file_name)
                                            #x_selected = x_valid_batch.astype(np.float16)
                                            x_selected = x_valid_batch
                                            valid_end_output = model.predict(x_selected)
                                            sum_valid_end_output = np.sum(valid_end_output, axis=0) #1xnClass
                                            final_layer_stored_matrix[nFileValid,:] = sum_valid_end_output
                                        np.savez(final_layer_file, matrix=final_layer_stored_matrix, file_name=file_name_list)

                                        #----------------------------------------- Extract embbeding
                                        #---------- generate visual model
                                        for layer in model.layers:
                                            #if layer.name == 'global_max_pooling2d':
                                            if layer.name == 'dense':
                                                successive_outputs = layer.output                   #successive_outputs = [layer.output for layer in model.layers[1:]]
                                                #print(layer.name)
                                            #else:
                                                #print(layer.name)
                                        visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)

                                        #---------- extract from train dir
                                        final_layer_res_dir = os.path.abspath(os.path.join(stored_dir, "03_train_emb"))
                                        if not os.path.exists(final_layer_res_dir):
                                            os.makedirs(final_layer_res_dir)

                                        for nFileValid in range(0,n_train):
                                            x_valid_batch, valid_file_name = generator_ins.get_single_file(nFileValid, args.train_dir)
                                            #x_selected = x_valid_batch.astype(np.float16)
                                            x_selected = x_valid_batch
                                            embedding = visualization_model.predict(x_selected)
                                            #print(np.shape(embedding), valid_file_name)
                                            final_layer_file = os.path.abspath(os.path.join(final_layer_res_dir, valid_file_name.split('.')[0]))
                                            np.save(final_layer_file, embedding)
                                        print('============================')

                                        #---------- extract from test dir
                                        final_layer_res_dir = os.path.abspath(os.path.join(stored_dir, "03_test_emb"))
                                        if not os.path.exists(final_layer_res_dir):
                                            os.makedirs(final_layer_res_dir)

                                        for nFileValid in range(0,n_valid):
                                            x_valid_batch, valid_file_name = generator_ins.get_single_file(nFileValid, args.test_dir)
                                            x_selected = x_valid_batch
                                            #x_selected = x_valid_batch.astype(np.float16)
                                            embedding = visualization_model.predict(x_selected)
                                            #print(np.shape(embedding), valid_file_name)
                                            final_layer_file = os.path.abspath(os.path.join(final_layer_res_dir, valid_file_name.split('.')[0]))
                                            np.save(final_layer_file, embedding)
                                        print('============================')
                                        exit()


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
