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
from tensorflow.keras.applications import EfficientNetB7
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
from tensorflow.keras import layers

#----- generator
#sys.path.append('./02_models/')
#from test_model import *
from generator import *
from hypara_model import *

#--------------------------------------------------------------------------------------
class Linear(layers.Layer):
    def __init__(self, units=512):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w1 = self.add_weight(
            name='w1',
            shape=(self.units,),
            initializer="random_normal",
            trainable=True,
        )
        self.w2 = self.add_weight(
            name='w2',
            shape=(self.units,),
            initializer="random_normal",
            trainable=True,
        )
        self.w3 = self.add_weight(
            name='w3',
            shape=(self.units,),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            name='b',
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        #print(np.shape(inputs[0]))
        return tf.math.multiply(inputs[0], self.w1) + tf.math.multiply(inputs[1], self.w2) + tf.math.multiply(inputs[2], self.w3) + self.b
        #return tf.math.multiply(inputs[0], self.w1) + tf.math.multiply(inputs[1], self.w2) + self.b
        #return tf.math.multiply(inputs, self.w1) + self.b
        #return inputs
   

#------------------------------------------------------------------------------------------------------------ main
def main():
    print("\n ==================================================================== SETUP PARAMETERS...")
    print("-------- PARSER:")
    parser = init_argparse()
    args   = parser.parse_args()
    OUT_DIR          = args.out_dir

    EMB01_TRAIN      = args.emb01_train
    EMB01_TEST       = args.emb01_test

    EMB02_TRAIN      = args.emb02_train
    EMB02_TEST       = args.emb02_test

    EMB03_TRAIN      = args.emb03_train
    EMB03_TEST       = args.emb03_test

    IS_TRAINING      = args.is_training
    IS_TESTING       = args.is_testing
    IS_EXTRACT       = args.is_extract

    print("-------- Hyper parameters:")
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
            linear_layer = Linear(512)
            with open(os.path.join(stored_dir,"train_acc_log.txt"), "a") as text_file:
                text_file.write("---------- Latest Model Loaded from dir: {}\n".format(best_model_dir))
            model = tf.keras.models.load_model(best_model_file)
        else:
            with open(os.path.join(stored_dir,"train_acc_log.txt"), "a") as text_file:
                text_file.write("---------- Creat A New Model at Dir: {}\n".format(best_model_dir))
            input_shape = (512,)
            input01 = Input(shape=input_shape)
            input02 = Input(shape=input_shape)
            input03 = Input(shape=input_shape)
            linear_layer = Linear(512)
            #x = Linear([input01, input02])
            x = linear_layer([input01, input02, input03])
            #print(np.shape(x))

            x = Dropout(0.4)(x)
            predictions = Dense(45, 
                                activation='softmax',
                                kernel_initializer=initializers.TruncatedNormal(mean=0.0,stddev=0.1),
                                kernel_regularizer=regularizers.l2(1e-5),
                                bias_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                                bias_regularizer=regularizers.l2(1e-5)
                               )(x)

            model = Model(inputs=[input01, input02, input03], outputs=predictions)

        #-----------------------------------    
        model.summary()
        #exit()
        opt = tf.keras.optimizers.Adam(LEARNING_RATE)
        #model.compile(loss="kullback_leibler_divergence", optimizer=opt, metrics=["accuracy"])   
        model.compile(loss=tf.keras.losses.kullback_leibler_divergence, optimizer=opt, metrics=["accuracy"])   

    old_file_valid_acc  = 0
    old_eva_acc_avg     = 0
    test_threshold      = 0.99

    #---Generator
    generator_ins    = generator(EMB01_TRAIN, EMB01_TEST, EMB02_TRAIN, EMB02_TEST, EMB03_TRAIN, EMB03_TEST)

    n_train          = generator_ins.get_file_num(EMB01_TRAIN)
    batch_num        = generator_ins.get_batch_num(BATCH_SIZE, EMB01_TRAIN)

    n_valid          = generator_ins.get_file_num(EMB01_TEST)
    batch_test       = 100
    batch_num_test   = generator_ins.get_batch_num(batch_test, EMB01_TEST)

    if IS_TRAINING == 'yes':
        for nEpoch in range(NUM_EPOCHS):
            print("\n=======================  Epoch is", nEpoch, ";============================")
            for nBatchTrain in range(START_BATCH, batch_num):
                x_emb01_batch, x_emb02_batch, x_emb03_batch, y_train_batch = generator_ins.get_batch(nBatchTrain, BATCH_SIZE, IS_MIXUP, True) #    
                #print(np.shape(x_emb01_batch), np.shape(x_emb02_batch))
                #exit()
                [train_loss, train_acc] = model.train_on_batch([x_emb01_batch, x_emb02_batch, x_emb03_batch], y_train_batch,reset_metrics=True)
                if (nBatchTrain % CHECKPOINT_EVERY == 0):  
                    print("Epoch: {}, TRAIN Accuracy:{}".format(nEpoch,train_acc))
                    with open(os.path.join(stored_dir,"train_acc_log.txt"), "a") as text_file:
                        text_file.write("Epoch:{}, TRAIN ACC:{} \n".format(nEpoch, train_acc))

            if(train_acc >= test_threshold) and nEpoch > 10:   
                if IS_TESTING == 'yes':  
                    eva_acc_avg = 0
                    for nBatchEva in range(0,batch_num_test):
                        x_eva01_batch, x_eva02_batch, x_eva03_batch, y_eva_batch = generator_ins.get_batch(nBatchEva, batch_test, False, False) #NO MIXUP FOR EVA    
                        [eva_loss, eva_acc] = model.evaluate([x_eva01_batch, x_eva02_batch, x_eva03_batch], y_eva_batch, verbose=0)
                        eva_acc_avg += eva_acc
                    eva_acc_avg = eva_acc_avg/batch_num_test

                    #01/ Save model when success evaluating on batches
                    if (eva_acc_avg > old_eva_acc_avg):
                        old_eva_acc_avg  = eva_acc_avg 
                        model.save(best_model_file)
                        with open(os.path.join(stored_dir,"train_acc_log.txt"), "a") as text_file:
                            text_file.write("Save best model at Epoch: {}; Score {}\n".format(nEpoch, old_eva_acc_avg))

    #------------------------------ EXTRACTION
    if (IS_EXTRACT == 'yes'):
        ### #---------------------------------------- Extract Prob
        ### final_layer_stored_matrix = np.zeros([n_valid, N_CLASS]) #file_num x nClass
    
        ### final_layer_res_dir = os.path.abspath(os.path.join(stored_dir, "01_pre_res"))
        ### if not os.path.exists(final_layer_res_dir):
        ###     os.makedirs(final_layer_res_dir)
        ### final_layer_file = os.path.abspath(os.path.join(final_layer_res_dir, "pre_train_res"))
        ### file_name_list = []       
        ### for nFileValid in range(0,n_valid):
        ###     x_valid01_batch, x_valid02_batch, x_valid03_batch, valid_file_name = generator_ins.get_single_file(nFileValid, EMB01_TEST, EMB02_TEST, EMB03_TEST,)
    
        ###     file_name_list.append(valid_file_name)
        ###     #x_selected = x_valid_batch.astype(np.float16)
        ###     valid_end_output = model.predict([x_valid01_batch, x_valid02_batch, x_valid03_batch])
        ###     sum_valid_end_output = np.sum(valid_end_output, axis=0) #1xnClass
        ###     final_layer_stored_matrix[nFileValid,:] = sum_valid_end_output
        ### np.savez(final_layer_file, matrix=final_layer_stored_matrix, file_name=file_name_list)

        #----------------------------------------- Extract embbeding
        #---------- generate visual model
        for layer in model.layers:
            #if layer.name == 'global_max_pooling2d':
            if layer.name == 'linear':
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
            x_valid01_batch, x_valid02_batch, x_valid03_batch, valid_file_name = generator_ins.get_single_file(nFileValid, EMB01_TRAIN, EMB02_TRAIN, EMB03_TRAIN)
            #x_selected = x_valid_batch.astype(np.float16)
            embedding = visualization_model.predict([x_valid01_batch, x_valid02_batch, x_valid03_batch])
            #print(np.shape(embedding), valid_file_name)
            final_layer_file = os.path.abspath(os.path.join(final_layer_res_dir, valid_file_name.split('.')[0]))
            np.save(final_layer_file, embedding)
        print('============================')

        #---------- extract from test dir
        final_layer_res_dir = os.path.abspath(os.path.join(stored_dir, "03_test_emb"))
        if not os.path.exists(final_layer_res_dir):
            os.makedirs(final_layer_res_dir)

        for nFileValid in range(0,n_valid):
            x_valid01_batch, x_valid02_batch, x_valid03_batch, valid_file_name = generator_ins.get_single_file(nFileValid, EMB01_TEST, EMB02_TEST, EMB03_TEST)
            #x_selected = x_valid_batch.astype(np.float16)
            embedding = visualization_model.predict([x_valid01_batch, x_valid02_batch, x_valid03_batch])
            final_layer_file = os.path.abspath(os.path.join(final_layer_res_dir, valid_file_name.split('.')[0]))
            np.save(final_layer_file, embedding)

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
        "--emb01_train", required=True,
        help=' --train_dir <train spectrogram directory>'
    )
    parser.add_argument(
        "--emb01_test", required=True,
        help=' --test_dir <test spectrogram directory>'
    )
    parser.add_argument(
        "--emb02_train", required=True,
        help=' --train_dir <train spectrogram directory>'
    )
    parser.add_argument(
        "--emb02_test", required=True,
        help=' --test_dir <test spectrogram directory>'
    )
    parser.add_argument(
        "--emb03_train", required=True,
        help=' --train_dir <train spectrogram directory>'
    )
    parser.add_argument(
        "--emb03_test", required=True,
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
