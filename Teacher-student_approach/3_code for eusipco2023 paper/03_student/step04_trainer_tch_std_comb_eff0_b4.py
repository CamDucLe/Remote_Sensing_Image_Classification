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
import tensorflow as tf
from sklearn import datasets, svm, metrics
from tensorflow.keras.applications import DenseNet121, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling1D
from tensorflow.keras.layers import BatchNormalization


#----- generator
sys.path.append('./02_models/')
#from baseline import *
from model_cnn_bs import *
from re import *
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
    IS_EXTRACTING    = args.is_extracting

    print("-------- Hyper parameters:")
    NF_AUD           = hypara().nF_aud  
    NT_AUD           = hypara().nT_aud
    NC_AUD           = hypara().nC_aud

    BATCH_SIZE       = hypara().batch_size
    START_BATCH      = hypara().start_batch
    LEARNING_RATE    = hypara().learning_rate
    IS_AUG           = hypara().is_aug
    CHECKPOINT_EVERY = hypara().check_every
    N_CLASS          = hypara().class_num
    NUM_EPOCHS       = hypara().epoch_num
    
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


    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
    
        #Instance model or reload an available model
        if os.path.isfile(best_model_file):
            model = tf.keras.models.load_model(best_model_file)
            with open(os.path.join(stored_dir,"train_acc_log.txt"), "a") as text_file:
                text_file.write("Latest model is loaded from: {} ...\n".format(best_model_dir))

            #model = baseline(nF_aud=NF_AUD, nT_aud=NT_AUD, nC_aud=NC_AUD, nH_img=NH_IMG, nW_img=NW_IMG, nC_img=NC_IMG, nClass=N_CLASS, chanDim=-1)
            #print(len(model.layers))
            #for nLayer in range(0, len(model1.layers)):
            #    if re.search('conv2d', model1.layers[nLayer].name):
            #        print('Prunning weight of: ', model1.layers[nLayer].name)
            #        #weight = np.array(model1.layers[nLayer].get_weights())    #weigh is a list
            #        weights = model1.layers[nLayer].trainable_weights
            #        #print(np.shape(weights))
            #        new_weight = []
            #        for index, weight_matrix in enumerate(weights):
            #            weight_mem = weight_matrix.numpy()
            #            critical_value = np.percentile(np.abs(weight_mem), 35)
            #            mask_indices = np.abs(weight_mem) < critical_value
            #            weight_mem[mask_indices] = 0
            #            new_weight.append(weight_mem)
            #        model.layers[nLayer].set_weights(new_weight)
            #    else:
            #        print('Keep weight of: ', model1.layers[nLayer].name)
            #        model.layers[nLayer].set_weights(model1.layers[nLayer].get_weights())
                     
        else:
            base_model = EfficientNetB0(weights='imagenet', include_top=False)
            #x = base_model.output
            x = base_model.get_layer('block4c_add').output
            x = GlobalAveragePooling2D()(x)
            x = Dropout(0.4)(x)

            y = Dense(512,
                      activation='relu',
                      kernel_initializer=initializers.TruncatedNormal(mean=0.0,stddev=0.1),
                      kernel_regularizer=regularizers.l2(1e-5),
                      bias_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                      bias_regularizer=regularizers.l2(1e-5)
                     )(x)
            x = Dropout(0.4)(y)

            predictions = Dense(45,
                                activation='softmax',
                                kernel_initializer=initializers.TruncatedNormal(mean=0.0,stddev=0.1),
                                kernel_regularizer=regularizers.l2(1e-5),
                                bias_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                                bias_regularizer=regularizers.l2(1e-5)
                               )(x)

            model = Model(inputs=base_model.input, outputs=[predictions,y])

            with open(os.path.join(stored_dir,"train_acc_log.txt"), "a") as text_file:
                text_file.write("New model instance is created...\n")
        model.summary()
        #exit()

        ## initialize the optimizer and compile the model
        set_opt = tf.keras.optimizers.Adam(LEARNING_RATE)
        set_loss = {'dense_1':tf.keras.losses.categorical_crossentropy, 'dense':tf.keras.losses.mean_squared_error}
        set_loss_weights={'dense_1': 1.0, 'dense': 1.0},
        set_metric = {'dense_1':'accuracy', 'dense':'mean_squared_error'}

        model.compile(loss=set_loss, loss_weights=set_loss_weights, optimizer=set_opt, metrics=set_metric)


        old_eva_acc_avg = 0
        test_threshold  = 0.97
        generator_ins   = generator(args.train_dir, args.eva_dir, args.train_emb_dir, args.eva_emb_dir)
        n_valid         = generator_ins.get_file_num(generator_ins.eva_dir)
        batch_num       = generator_ins.get_batch_num(BATCH_SIZE, args.train_dir)

        batch_test = 100
        batch_num_test  = generator_ins.get_batch_num(batch_test, args.eva_dir)
        ##print(batch_num)
        ##exit()

        if IS_TRAINING == 'yes':
            for nEpoch in range(NUM_EPOCHS):
                generator_ins.random_id(nEpoch)

                for nBatchTrain in range(START_BATCH, batch_num):
                    #x_train_batch, y_train_batch, n_image = generator_ins.get_batch(nBatchTrain, BATCH_SIZE, IS_AUG, True) #    
                    #[train_loss, train_acc] = model.train_on_batch(x_train_batch, y_train_batch, reset_metrics=True)  #one output
                    x_train_batch, y_train_batch, n_image, y_emb = generator_ins.get_batch(nBatchTrain, BATCH_SIZE, IS_AUG, True) #
                    [sum_loss, loss1, loss2, train_acc, train_acc_emb] = model.train_on_batch(x_train_batch, [y_train_batch, y_emb], reset_metrics=True)  #one output

                    if (nBatchTrain % CHECKPOINT_EVERY == 0):  
                        print("Epoch: {}, TRAIN Accuracy:{}".format(nEpoch, train_acc))
                        with open(os.path.join(stored_dir,"train_acc_log.txt"), "a") as text_file:
                            text_file.write("Epoch:{}, TRAIN ACC:{} \n".format(nEpoch, train_acc))

                if (train_acc >= test_threshold) and (nEpoch >10):   
                    if IS_TESTING == 'yes':  
                        eva_acc_avg = 0
                        for nBatchEva in range(0,batch_num_test):
                            x_eva_batch, y_eva_batch, n_image, y_eva_emb = generator_ins.get_batch(nBatchEva, batch_test, False, False) #NO MIXUP FOR EVA    
                            [eva_all_loss, eva_loss1, eva_loss2, eva_acc, eva_acc_emb] = model.evaluate(x_eva_batch, [y_eva_batch, y_eva_emb], verbose=0)
                            eva_acc_avg += eva_acc
                        eva_acc_avg = eva_acc_avg/batch_num_test

                        if eva_acc_avg > old_eva_acc_avg:
                             old_eva_acc_avg = eva_acc_avg
                             model.save(best_model_file)
                             with open(os.path.join(stored_dir,"train_acc_log.txt"), "a") as text_file:
                                 text_file.write("Save best model at Epoch: {}; TRAIN ACC:{}; EVA ACC:{}\n".format(nEpoch, train_acc, eva_acc_avg))

def update_lr(model, epoch):
    if epoch >= 100:
        new_lr = 0.0001
        model.optimizer.lr.assign(new_lr)

def cus_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true-y_pred))


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
        "--eva_dir", required=True,
        help=' --eva_dir <eva spectrogram directory>'
    )
    parser.add_argument(
        "--train_emb_dir", required=True,
        help=' --eva_dir <eva spectrogram directory>'
    )
    parser.add_argument(
        "--eva_emb_dir", required=True,
        help=' --eva_dir <eva spectrogram directory>'
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
        "--is_extracting", required=True,
        help=' --is_extracting<yes/no>'
    )
    return parser


if __name__ == "__main__":
    main()
