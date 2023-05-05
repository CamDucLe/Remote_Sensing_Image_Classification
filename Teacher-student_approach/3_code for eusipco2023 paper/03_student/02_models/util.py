#-------------------- Tensoflow packages
from tensorflow.keras import initializers
from tensorflow.keras import regularizers

#---
from tensorflow.keras.layers import Conv2D, Conv1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D   
from tensorflow.keras.layers import GlobalMaxPooling2D   
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Add

from tensorflow.keras.layers import Reshape, multiply, Permute, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.activations import sigmoid
import numpy as np
import tensorflow as tf

#-------------------------------------------------------------------------------------------------- CBAM
def l_normalize(x, axes_sel):
    gl_mean, gl_var = tf.nn.moments(x, axes = axes_sel, keepdims=True)
    x_std = (x - gl_mean)/tf.math.sqrt(gl_var)
    x_nor = 0.4*x + x_std
    return x_nor

def attach_attention_module(net, attention_module):
  if attention_module == 'se_block': # SE_block
    net = se_block(net)
  elif attention_module == 'cbam_block': # CBAM_block
    net = cbam_block(net)
  else:
    raise Exception("'{}' is not supported attention module!".format(attention_module))

  return net

def se_block(input_feature, ratio=8):
	"""Contains the implementation of Squeeze-and-Excitation(SE) block.
	As described in https://arxiv.org/abs/1709.01507.
	"""
	
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        
	channel = input_feature._shape_val[channel_axis]

	se_feature = GlobalAveragePooling2D()(input_feature)
	se_feature = Reshape((1, 1, channel))(se_feature)
	assert se_feature._shape_val[1:] == (1,1,channel)
	se_feature = Dense(channel // ratio,
					   activation='relu',
					   #kernel_initializer='he_normal',
					   kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1), #lampham
					   use_bias=True,
					   #bias_initializer='zeros')(se_feature)
					   bias_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1))(se_feature) #lampham
	assert se_feature._shape_val[1:] == (1,1,channel//ratio)
	se_feature = Dense(channel,
					   activation='sigmoid',
					   kernel_initializer='he_normal',
					   use_bias=True,
					   #bias_initializer='zeros')(se_feature)
					   bias_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1))(se_feature) # lampham
	assert se_feature._shape_val[1:] == (1,1,channel)
	if K.image_data_format() == 'channels_first':
		se_feature = Permute((3, 1, 2))(se_feature)

	se_feature = multiply([input_feature, se_feature])
	return se_feature

def cbam_block(cbam_feature, ratio=8):
	"""Contains the implementation of Convolutional Block Attention Module(CBAM) block.
	As described in https://arxiv.org/abs/1807.06521.
	"""
	
	cbam_feature = channel_attention(cbam_feature, ratio)
        #print(np.shape(cbam_feature))
	cbam_feature = spatial_attention(cbam_feature)
        #print(np.shape(cbam_feature))
        #exit()
	return cbam_feature

def channel_attention(input_feature, ratio=8):
	
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    #channel_axis = -1  #lampham
    #channel = input_feature._shape_val[channel_axis]
    channel = input_feature._shape_val[channel_axis]
    print(channel)
    
    shared_layer_one = Dense(channel//ratio,
                             activation='relu',
                             #kernel_initializer='he_normal',
                             kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1), #lampham
                             use_bias=True,
                             #bias_initializer='zeros')
                             bias_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1)) #lampham

    shared_layer_two = Dense(channel,
                             #kernel_initializer='he_normal',
                             kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1), #lampham
                             use_bias=True,
                             #bias_initializer='zeros')
                             bias_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1)) #lampham
    
    avg_pool = GlobalAveragePooling2D()(input_feature)    
    avg_pool = Reshape((1,1,channel))(avg_pool)
    assert avg_pool._shape_val[1:] == (1,1,channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool._shape_val[1:] == (1,1,channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool._shape_val[1:] == (1,1,channel)
    
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1,1,channel))(max_pool)
    assert max_pool._shape_val[1:] == (1,1,channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool._shape_val[1:] == (1,1,channel//ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool._shape_val[1:] == (1,1,channel)
    
    cbam_feature = Add()([avg_pool,max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    
    if K.image_data_format() == "channels_first":
    	cbam_feature = Permute((3, 1, 2))(cbam_feature)
    
    return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature):
	kernel_size = 7
	
	if K.image_data_format() == "channels_first":
		channel = input_feature._shape_val[1]
		cbam_feature = Permute((2,3,1))(input_feature)
	else:
		channel = input_feature._shape_val[-1]
		cbam_feature = input_feature
	
	avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
	assert avg_pool._shape_val[-1] == 1
	max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
	assert max_pool._shape_val[-1] == 1
	concat = Concatenate(axis=3)([avg_pool, max_pool])
	assert concat._shape_val[-1] == 2
	cbam_feature = Conv2D(filters = 1,
					kernel_size=kernel_size,
					strides=1,
					padding='same',
					activation='sigmoid',
					#kernel_initializer='he_normal',
	     	                        kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1), #lampham
					use_bias=False)(concat)	
	assert cbam_feature._shape_val[-1] == 1
	
	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2))(cbam_feature)
		
	return multiply([input_feature, cbam_feature])

#-------------------- Layer definition
def l_batch_norm(x, bn_moment=0.9, bn_dim=-1):
    return BatchNormalization(axis     = bn_dim,
                              momentum = bn_moment
                             )(x)

def l_conv2d(x, conv2d_filters, conv2d_kernel, conv2d_padding='same', conv2d_dilation_rate=(1,1)):
    return  Conv2D(filters       = conv2d_filters, 
                   kernel_size   = conv2d_kernel, 
                   padding       = conv2d_padding,
                   dilation_rate = conv2d_dilation_rate,

                   kernel_initializer=initializers.TruncatedNormal(mean=0.0,stddev=0.1),
                   kernel_regularizer=regularizers.l2(1e-5),
                   bias_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                   bias_regularizer=regularizers.l2(1e-5)
                  )(x)

def l_act(x, act_type, name=''):
    return Activation(act_type, name=name)(x)

def l_avg_pool2d(x, pool_size=(2,2), pool_strides=(2,2), pool_padding='valid'):
    return AveragePooling2D(pool_size = pool_size, 
                            strides   = pool_strides,
                            padding   = pool_padding
                           )(x)

def l_max_pool2d(x, pool_size=(2,2), pool_strides=(2,2), pool_padding='valid'):
    return MaxPooling2D(pool_size = pool_size, 
                        strides   = pool_strides,
                        padding   = pool_padding
                       )(x)

def l_dense(x, den_unit, name_dense):
    return Dense(units = den_unit,

                 kernel_initializer=initializers.TruncatedNormal(mean=0.0,stddev=0.1),
                 kernel_regularizer=regularizers.l2(1e-5),
                 bias_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                 bias_regularizer=regularizers.l2(1e-5),
                 name=name_dense
                )(x)

def l_drop(x, drop_rate):
    return Dropout(drop_rate)(x)

def l_glo_avg_pool2d(x):
    return GlobalAveragePooling2D()(x)

def l_glo_max_pool2d(x):
    return GlobalMaxPooling2D()(x)

def sig_conv2d(x,
               conv2d_filters, conv2d_kernel, conv2d_padding='same', conv2d_dilation_rate=(1,1), 
               is_bn=True, bn_moment=0.9, bn_dim=-1,
               act_type='relu', 
               is_pool=True, pool_type='AP', pool_size=(2,2), pool_strides=(2,2), pool_padding='valid',
               is_drop=True,drop_rate=0.1
              ):

    #batchnorm
    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    #conv
    x = l_conv2d(x, 
                 conv2d_filters=conv2d_filters, 
                 conv2d_kernel=conv2d_kernel,
                 conv2d_padding=conv2d_padding,
                 conv2d_dilation_rate=conv2d_dilation_rate
                )
    #act
    x = l_act(x, act_type)
    #batchnorm
    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    #pooling
    if(is_pool):
        if pool_type == 'AP':
            x = l_avg_pool2d(x, pool_size=pool_size, pool_strides=pool_strides, pool_padding=pool_padding)
        elif pool_type == 'MP':
            x = l_max_pool2d(x, pool_size=pool_size, pool_strides=pool_strides, pool_padding=pool_padding)
        elif pool_type == 'GAP':
            x = l_glo_avg_pool2d(x)
        elif pool_type == 'G_MIX':
            x1 = l_glo_avg_pool2d(x)
            x2 = l_glo_max_pool2d(x)
            x  = x1 + x2
    #drop
    if is_drop:
        x = l_drop(x, drop_rate=drop_rate)

    return x

def dob_conv2d(x,
               conv2d_filters, conv2d_kernel, conv2d_padding='same', conv2d_dilation_rate=(1,1), 
               is_bn=True, bn_moment=0.9, bn_dim=-1,
               act_type='relu', 
               is_pool=True, pool_type='AP', pool_size=(2,2), pool_strides=(2,2), pool_padding='valid',
               is_drop=True, drop_rate=0.1
              ):

    #batchnorm 00
    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    #conv 00
    x = l_conv2d(x, 
                 conv2d_filters=conv2d_filters, 
                 conv2d_kernel=conv2d_kernel,
                 conv2d_padding=conv2d_padding,
                 conv2d_dilation_rate=conv2d_dilation_rate
                )
    #act 00
    x = l_act(x, act_type)

    #batchnorm 01
    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    #conv 01    
    x = l_conv2d(x, 
                 conv2d_filters=conv2d_filters, 
                 conv2d_kernel=conv2d_kernel,
                 conv2d_padding=conv2d_padding,
                 conv2d_dilation_rate=conv2d_dilation_rate
                )
    #act 01
    x = l_act(x, act_type)

    #batchnorm 02
    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)

    #pooling
    if(is_pool):
        if pool_type == 'AP':
            x = l_avg_pool2d(x, pool_size=pool_size, pool_strides=pool_strides, pool_padding=pool_padding)
        elif pool_type == 'MP':
            x = l_max_pool2d(x, pool_size=pool_size, pool_strides=pool_strides, pool_padding=pool_padding)
        elif pool_type == 'GAP':
            x = l_glo_avg_pool2d(x)
        elif pool_type == 'G_MIX':
            x1 = l_glo_avg_pool2d(x)
            x2 = l_glo_max_pool2d(x)
            x  = x1 + x2
    #drop
    if is_drop:
        x = l_drop(x, drop_rate=drop_rate)

    return x



def l_inc01(layer_in, f1, f2, f3, f4):
    # 1x1 conv
    conv11 = Conv2D(f1, (1,1), padding='same', activation='relu',
                   dilation_rate      = (1,1),
                   kernel_initializer = initializers.TruncatedNormal(mean=0.0,stddev=0.1),
                   kernel_regularizer = regularizers.l2(1e-5),
                   bias_initializer   = initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                   bias_regularizer   = regularizers.l2(1e-5)
                  )(layer_in)
    # 3x3 conv
    conv33 = Conv2D(f2, (3,3), padding='same', activation='relu',
                   dilation_rate      = (1,1),
                   kernel_initializer = initializers.TruncatedNormal(mean=0.0,stddev=0.1),
                   kernel_regularizer = regularizers.l2(1e-5),
                   bias_initializer   = initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                   bias_regularizer   = regularizers.l2(1e-5)
                  )(layer_in)
    # 5x5 conv
    conv14 = Conv2D(f3, (1,4), padding='same', activation='relu',
                   dilation_rate      = (1,1),
                   kernel_initializer = initializers.TruncatedNormal(mean=0.0,stddev=0.1),
                   kernel_regularizer = regularizers.l2(1e-5),
                   bias_initializer   = initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                   bias_regularizer   = regularizers.l2(1e-5)
                  )(layer_in)

    # 3x3 max pooling
    pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
    pool_conv11 = Conv2D(f4, (1,1), padding='same', activation='relu',
                   dilation_rate      = (1,1),
                   kernel_initializer = initializers.TruncatedNormal(mean=0.0,stddev=0.1),
                   kernel_regularizer = regularizers.l2(1e-5),
                   bias_initializer   = initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                   bias_regularizer   = regularizers.l2(1e-5)
                  )(pool)

    # concatenate filters, assumes filters/channels last
    layer_out = Concatenate(axis=-1)([conv11, conv33, conv14, pool_conv11])

    return layer_out

def l_inc03_03(layer_in, f1, f2, f3, f4):
    # 1x1 conv
    conv11 = Conv2D(f1, (1,1), padding='same', activation='relu',
                   dilation_rate      = (1,1),
                   kernel_initializer = initializers.TruncatedNormal(mean=0.0,stddev=0.1),
                   kernel_regularizer = regularizers.l2(1e-5),
                   bias_initializer   = initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                   bias_regularizer   = regularizers.l2(1e-5)
                  )(layer_in)
    # 3x3 conv
    conv33 = Conv2D(f2, (3,3), padding='same', activation='relu',
                   dilation_rate      = (1,1),
                   kernel_initializer = initializers.TruncatedNormal(mean=0.0,stddev=0.1),
                   kernel_regularizer = regularizers.l2(1e-5),
                   bias_initializer   = initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                   bias_regularizer   = regularizers.l2(1e-5)
                  )(layer_in)


    #conv33 = l_dec_conv2d(layer_in, (f1+f2+f3+f4)/2, f2, (3,3))

    ## 5x5 conv
    conv14 = Conv2D(f3, (5,5), padding='same', activation='relu',
                   dilation_rate      = (1,1),
                   kernel_initializer = initializers.TruncatedNormal(mean=0.0,stddev=0.1),
                   kernel_regularizer = regularizers.l2(1e-5),
                   bias_initializer   = initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                   bias_regularizer   = regularizers.l2(1e-5)
                  )(layer_in)

    #conv14 = l_dec_conv2d(layer_in, (f1+f2+f3+f4)/2, f3, (5,5))

    # 3x3 max pooling
    pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
    pool_conv11 = Conv2D(f4, (1,1), padding='same', activation='relu',
                   dilation_rate      = (1,1),
                   kernel_initializer = initializers.TruncatedNormal(mean=0.0,stddev=0.1),
                   kernel_regularizer = regularizers.l2(1e-5),
                   bias_initializer   = initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                   bias_regularizer   = regularizers.l2(1e-5)
                  )(pool)

    # concatenate filters, assumes filters/channels last
    layer_out = Concatenate(axis=-1)([conv11, conv33, conv14, pool_conv11])

    return layer_out

def l_inc03_02(layer_in, f1, f2, f3, f4):
    # 1x1 conv
    conv11 = Conv2D(f1, (1,1), padding='same', activation='relu',
                   dilation_rate      = (1,1),
                   kernel_initializer = initializers.TruncatedNormal(mean=0.0,stddev=0.1),
                   kernel_regularizer = regularizers.l2(1e-5),
                   bias_initializer   = initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                   bias_regularizer   = regularizers.l2(1e-5)
                  )(layer_in)
    # 3x3 conv
    conv33 = Conv2D(f2, (1,3), padding='same', activation='relu',
                   dilation_rate      = (1,1),
                   kernel_initializer = initializers.TruncatedNormal(mean=0.0,stddev=0.1),
                   kernel_regularizer = regularizers.l2(1e-5),
                   bias_initializer   = initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                   bias_regularizer   = regularizers.l2(1e-5)
                  )(layer_in)
    # 5x5 conv
    conv14 = Conv2D(f3, (1,5), padding='same', activation='relu',
                   dilation_rate      = (1,1),
                   kernel_initializer = initializers.TruncatedNormal(mean=0.0,stddev=0.1),
                   kernel_regularizer = regularizers.l2(1e-5),
                   bias_initializer   = initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                   bias_regularizer   = regularizers.l2(1e-5)
                  )(layer_in)

    # 3x3 max pooling
    pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
    pool_conv11 = Conv2D(f4, (1,1), padding='same', activation='relu',
                   dilation_rate      = (1,1),
                   kernel_initializer = initializers.TruncatedNormal(mean=0.0,stddev=0.1),
                   kernel_regularizer = regularizers.l2(1e-5),
                   bias_initializer   = initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                   bias_regularizer   = regularizers.l2(1e-5)
                  )(pool)

    # concatenate filters, assumes filters/channels last
    layer_out = Concatenate(axis=-1)([conv11, conv33, conv14, pool_conv11])

    return layer_out

def l_inc03_01(layer_in, f1, f2, f3, f4):
    # 1x1 conv
    conv11 = Conv2D(f1, (1,1), padding='same', activation='relu',
                   dilation_rate      = (1,1),
                   kernel_initializer = initializers.TruncatedNormal(mean=0.0,stddev=0.1),
                   kernel_regularizer = regularizers.l2(1e-5),
                   bias_initializer   = initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                   bias_regularizer   = regularizers.l2(1e-5)
                  )(layer_in)
    # 3x3 conv
    conv33 = Conv2D(f2, (3,1), padding='same', activation='relu',
                   dilation_rate      = (1,1),
                   kernel_initializer = initializers.TruncatedNormal(mean=0.0,stddev=0.1),
                   kernel_regularizer = regularizers.l2(1e-5),
                   bias_initializer   = initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                   bias_regularizer   = regularizers.l2(1e-5)
                  )(layer_in)
    # 5x5 conv
    conv14 = Conv2D(f3, (5,1), padding='same', activation='relu',
                   dilation_rate      = (1,1),
                   kernel_initializer = initializers.TruncatedNormal(mean=0.0,stddev=0.1),
                   kernel_regularizer = regularizers.l2(1e-5),
                   bias_initializer   = initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                   bias_regularizer   = regularizers.l2(1e-5)
                  )(layer_in)

    # 3x3 max pooling
    pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
    pool_conv11 = Conv2D(f4, (1,1), padding='same', activation='relu',
                   dilation_rate      = (1,1),
                   kernel_initializer = initializers.TruncatedNormal(mean=0.0,stddev=0.1),
                   kernel_regularizer = regularizers.l2(1e-5),
                   bias_initializer   = initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                   bias_regularizer   = regularizers.l2(1e-5)
                  )(pool)

    # concatenate filters, assumes filters/channels last
    layer_out = Concatenate(axis=-1)([conv11, conv33, conv14, pool_conv11])

    return layer_out

def l_inc02(layer_in, f1, f2, f3, f4):
    # 1x1 conv
    conv11 = Conv2D(f1, (1,1), padding='same', activation='relu',
                   dilation_rate      = (1,1),
                   kernel_initializer = initializers.TruncatedNormal(mean=0.0,stddev=0.1),
                   kernel_regularizer = regularizers.l2(1e-4),
                   use_bias=True,
                   bias_initializer   = initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                   bias_regularizer   = regularizers.l2(1e-4)
                  )(layer_in)
    # 3x3 conv
    conv33 = Conv2D(f2, (3,3), padding='same', activation='relu',
                   dilation_rate      = (1,1),
                   kernel_initializer = initializers.TruncatedNormal(mean=0.0,stddev=0.1),
                   kernel_regularizer = regularizers.l2(1e-4),
                   use_bias=True,
                   bias_initializer   = initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                   bias_regularizer   = regularizers.l2(1e-4)
                  )(layer_in)
    # 5x5 conv
    conv14 = Conv2D(f3, (4,1), padding='same', activation='relu',
                   dilation_rate      = (1,1),
                   kernel_initializer = initializers.TruncatedNormal(mean=0.0,stddev=0.1),
                   kernel_regularizer = regularizers.l2(1e-4),
                   use_bias=True,
                   bias_initializer   = initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                   bias_regularizer   = regularizers.l2(1e-4)
                  )(layer_in)

    # 3x3 max pooling
    pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
    pool_conv11 = Conv2D(f4, (1,1), padding='same', activation='relu',
                   dilation_rate      = (1,1),
                   kernel_initializer = initializers.TruncatedNormal(mean=0.0,stddev=0.1),
                   kernel_regularizer = regularizers.l2(1e-4),
                   use_bias=True,
                   bias_initializer   = initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                   bias_regularizer   = regularizers.l2(1e-4)
                  )(pool)

    # concatenate filters, assumes filters/channels last
    layer_out = Concatenate(axis=-1)([conv11, conv33, conv14, pool_conv11])

    return layer_out

def tri_inc01(x,
              f1, f2, f3, f4, 
              is_bn=True, bn_moment=0.9, bn_dim=-1,
              act_type='relu', 
              is_pool=True, pool_type='AP', pool_size=(2,2), pool_strides=(2,2), pool_padding='valid',
              is_drop=True, drop_rate=0.1
             ):

    #batchnorm 01
    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    #conv 01
    x = l_inc01(x, f1, f2, f3, f4)
    #batchnorm 01
    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    #act 01
    x = l_act(x, act_type)

    #batchnorm 02
    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    #conv 02    
    x = l_inc01(x, f1, f2, f3, f4)
    #batchnorm 02
    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    #act 02
    x = l_act(x, act_type)

    #batchnorm 03
    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    #conv 03    
    x = l_inc01(x, f1, f2, f3, f4)
    #batchnorm 03
    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    #act 03
    x = l_act(x, act_type)

    #pooling
    if(is_pool):
        if pool_type == 'AP':
            x = l_avg_pool2d(x, pool_size=pool_size, pool_strides=pool_strides, pool_padding=pool_padding)
        elif pool_type == 'MP':
            x = l_max_pool2d(x, pool_size=pool_size, pool_strides=pool_strides, pool_padding=pool_padding)
        elif pool_type == 'GAP':
            x = l_glo_avg_pool2d(x)
        elif pool_type == 'G_MIX':
            x1 = l_glo_avg_pool2d(x)
            x2 = l_glo_max_pool2d(x)
            x  = x1 + x2
    #drop
    if is_drop:
        x = l_drop(x, drop_rate=drop_rate)

    return x

def dob_inc01(x,
              f1, f2, f3, f4, 
              is_bn=True, bn_moment=0.9, bn_dim=-1,
              act_type='relu', 
              is_pool=True, pool_type='AP', pool_size=(2,2), pool_strides=(2,2), pool_padding='valid',
              is_drop=True, drop_rate=0.1
             ):

    #batchnorm 01
    #if(is_bn):    
    #    x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    #conv 01
    x = l_inc01(x, f1, f2, f3, f4)
    #batchnorm 01
    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    #act 01
    x = l_act(x, act_type)

    #batchnorm 02
    #if(is_bn):    
    #    x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    #conv 02    
    x = l_inc01(x, f1, f2, f3, f4)
    #batchnorm 02
    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    #act 02
    x = l_act(x, act_type)

    ##batchnorm 02

    #pooling
    if(is_pool):
        if pool_type == 'AP':
            x = l_avg_pool2d(x, pool_size=pool_size, pool_strides=pool_strides, pool_padding=pool_padding)
        elif pool_type == 'MP':
            x = l_max_pool2d(x, pool_size=pool_size, pool_strides=pool_strides, pool_padding=pool_padding)
        elif pool_type == 'GAP':
            x = l_glo_avg_pool2d(x)
        elif pool_type == 'G_MIX':
            x1 = l_glo_avg_pool2d(x)
            x2 = l_glo_max_pool2d(x)
            x  = x1 + x2
    #drop
    if is_drop:
        x = l_drop(x, drop_rate=drop_rate)

    return x


def dob_cbam_inc01(x,
              f1, f2, f3, f4, 
              is_bn=True, bn_moment=0.9, bn_dim=-1,
              act_type='relu', 
              is_pool=True, pool_type='AP', pool_size=(2,2), pool_strides=(2,2), pool_padding='valid',
              is_drop=True, drop_rate=0.1
             ):

    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    x = l_inc01(x, f1, f2, f3, f4)
    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    x = l_act(x, act_type)

    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    x = l_inc01(x, f1, f2, f3, f4)
    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)

    #attention layer
    x_skip = x
    x_skip = attach_attention_module(x_skip, 'cbam_block')

    x = Add()([x, x_skip])
    x = l_act(x, act_type)
    ##batchnorm 02

    #pooling
    if(is_pool):
        if pool_type == 'AP':
            x = l_avg_pool2d(x, pool_size=pool_size, pool_strides=pool_strides, pool_padding=pool_padding)
        elif pool_type == 'MP':
            x = l_max_pool2d(x, pool_size=pool_size, pool_strides=pool_strides, pool_padding=pool_padding)
        elif pool_type == 'GAP':
            x = l_glo_avg_pool2d(x)
        elif pool_type == 'G_MIX':
            x1 = l_glo_avg_pool2d(x)
            x2 = l_glo_max_pool2d(x)
            x  = x1 + x2
    #drop
    if is_drop:
        x = l_drop(x, drop_rate=drop_rate)

    return x

def tri_cbam_inc01(x,
              f1, f2, f3, f4, 
              is_bn=True, bn_moment=0.9, bn_dim=-1,
              act_type='relu', 
              is_pool=True, pool_type='AP', pool_size=(2,2), pool_strides=(2,2), pool_padding='valid',
              is_drop=True, drop_rate=0.1
             ):

    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    x = l_inc01(x, f1, f2, f3, f4)
    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    x = l_act(x, act_type)

    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    x = l_inc01(x, f1, f2, f3, f4)
    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    x = l_act(x, act_type)

    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    x = l_inc01(x, f1, f2, f3, f4)
    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)

    #attention layer
    x_skip = x
    x_skip = attach_attention_module(x_skip, 'cbam_block')

    x = Add()([x, x_skip])
    x = l_act(x, act_type)
    ##batchnorm 02

    #pooling
    if(is_pool):
        if pool_type == 'AP':
            x = l_avg_pool2d(x, pool_size=pool_size, pool_strides=pool_strides, pool_padding=pool_padding)
        elif pool_type == 'MP':
            x = l_max_pool2d(x, pool_size=pool_size, pool_strides=pool_strides, pool_padding=pool_padding)
        elif pool_type == 'GAP':
            x = l_glo_avg_pool2d(x)
        elif pool_type == 'G_MIX':
            x1 = l_glo_avg_pool2d(x)
            x2 = l_glo_max_pool2d(x)
            x  = x1 + x2
    #drop
    if is_drop:
        x = l_drop(x, drop_rate=drop_rate)

    return x

def tri_cbam_inc01_02(x,
              f1, f2, f3, f4, 
              is_bn=True, bn_moment=0.9, bn_dim=-1,
              act_type='relu', 
              is_pool=True, pool_type='AP', pool_size=(2,2), pool_strides=(2,2), pool_padding='valid',
              is_drop=True, drop_rate=0.1
             ):

    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    x = l_inc01(x, f1, f2, f3, f4)
    x = l_inc01(x, f1, f2, f3, f4)
    x = l_inc01(x, f1, f2, f3, f4)
    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)

    #attention layer
    x_skip = x
    x_skip = attach_attention_module(x_skip, 'cbam_block')

    x = Add()([x, x_skip])
    x = l_act(x, act_type)
    ##batchnorm 02

    #pooling
    if(is_pool):
        if pool_type == 'AP':
            x = l_avg_pool2d(x, pool_size=pool_size, pool_strides=pool_strides, pool_padding=pool_padding)
        elif pool_type == 'MP':
            x = l_max_pool2d(x, pool_size=pool_size, pool_strides=pool_strides, pool_padding=pool_padding)
        elif pool_type == 'GAP':
            x = l_glo_avg_pool2d(x)
        elif pool_type == 'G_MIX':
            x1 = l_glo_avg_pool2d(x)
            x2 = l_glo_max_pool2d(x)
            x  = x1 + x2
    #drop
    if is_drop:
        x = l_drop(x, drop_rate=drop_rate)

    return x

def res_inc03(x,
              f1, f2, f3, f4, 
              is_bn=True, bn_moment=0.9, bn_dim=-1,
              act_type='relu', 
              is_pool=True, pool_type='AP', pool_size=(2,2), pool_strides=(2,2), pool_padding='valid',
              is_drop=True, drop_rate=0.1
             ):
    #skip
    x_skip = x
    x_skip = l_conv2d(x_skip, conv2d_filters=f1+f2+f3+f4, conv2d_kernel=(1,1))
    if(is_bn):    
        x_skip = l_batch_norm(x_skip, bn_moment=bn_moment, bn_dim=bn_dim)
    x_skip = l_act(x_skip, act_type) 
    x_skip = l_avg_pool2d(x_skip, pool_size=(3,3), pool_strides=(1,1), pool_padding='same')
    x_skip = l_normalize(x_skip, [1,2])

    #main
    x = l_conv2d(x, conv2d_filters=(f1+f2+f3+f4)/2, conv2d_kernel=(1,1))
    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    x = l_act(x, act_type) 
    #x = l_normalize(x, [1,2])

    #----- 01
    x_01 = l_inc03_01(x, f1, f2, f3, f4)
    if(is_bn):    
        x_01 = l_batch_norm(x_01, bn_moment=bn_moment, bn_dim=bn_dim)
    #x_01 = l_act(x_01, act_type)
    #x_01 = l_avg_pool2d(x_01, pool_size=(3,1), pool_strides=(1,1), pool_padding='same')
    #x_01 = l_normalize(x_01, [2,3])

    x_01 = l_inc03_01(x_01, f1, f2, f3, f4)
    if(is_bn):    
        x_01 = l_batch_norm(x_01, bn_moment=bn_moment, bn_dim=bn_dim)
    x_01 = l_act(x_01, act_type)
    x_01 = l_avg_pool2d(x_01, pool_size=(3,1), pool_strides=(1,1), pool_padding='same')

    #freq 01
    #x_01_skip = x_01
    #print(np.shape(x_01_skip))
    #x_01_skip = attach_attention_module(x_01_skip, 'cbam_block')
    #x_01 = x_01 + x_01_skip
    #x_01 = l_act(x_01, act_type) 
    #x_01 = attach_attention_module(x_01, 'cbam_block')
    #x_01 = l_normalize(x_01, [1,2])

    #----- 02
    x_02 = l_inc03_02(x, f1, f2, f3, f4)
    if(is_bn):    
        x_02 = l_batch_norm(x_02, bn_moment=bn_moment, bn_dim=bn_dim)
    #x_02 = l_act(x_02, act_type)
    #x_02 = l_avg_pool2d(x_02, pool_size=(1,3), pool_strides=(1,1), pool_padding='same')
    #x_02 = l_normalize(x_02, [2,3])

    x_02 = l_inc03_02(x_02, f1, f2, f3, f4)
    if(is_bn):    
        x_02 = l_batch_norm(x_02, bn_moment=bn_moment, bn_dim=bn_dim)
    x_02 = l_act(x_02, act_type)
    x_02 = l_avg_pool2d(x_02, pool_size=(1,3), pool_strides=(1,1), pool_padding='same')

    #freq 02
    #x_02_skip = x_02
    #x_02_skip = attach_attention_module(x_02_skip, 'cbam_block')
    #x_02 = x_02 + x_02_skip
    #x_02 = l_act(x_02, act_type) 
    #x_02 = attach_attention_module(x_02, 'cbam_block')
    #x_02 = l_normalize(x_02, [1,2])

    #----- 03
    x_03 = l_inc03_03(x, f1, f2, f3, f4)
    if(is_bn):    
        x_03 = l_batch_norm(x_03, bn_moment=bn_moment, bn_dim=bn_dim)
    #x_03 = l_act(x_03, act_type)
    #x_03 = l_avg_pool2d(x_03, pool_size=(3,3), pool_strides=(1,1), pool_padding='same')
    #x_03 = l_normalize(x_03, [2,3])

    x_03 = l_inc03_03(x_03, f1, f2, f3, f4)
    if(is_bn):    
        x_03 = l_batch_norm(x_03, bn_moment=bn_moment, bn_dim=bn_dim)
    x_03 = l_act(x_03, act_type)
    x_03 = l_avg_pool2d(x_03, pool_size=(3,3), pool_strides=(1,1), pool_padding='same')

    #freq 03
    #x_03 = attach_attention_module(x_03, 'cbam_block')
    #x_03 = l_normalize(x_03, [1,2])
    

    #merge
    x = Add()([x_01, x_02, x_03, x_skip])
    x = l_act(x, act_type) 

    #----------------------------- conv1d
    #x_skip02 =  x
    ###Freq average
    #x_skip02 = tf.math.reduce_mean(x_skip02, axis=2)
    ##print(np.shape(x))
    ##exit()

    #x_skip02 = l_conv1d(x_skip02, f1+f2+f3+f4, 5)
    ##if(is_bn):    
    ##    x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    #x_skip02 = l_act(x_skip02, act_type)
    #x_skip02 = tf.expand_dims(x_skip02, axis=2)

    #------------------------------cbam
    #x_skip02 = x
    #x_skip02 = attach_attention_module(x_skip02, 'cbam_block')
    #x = Add()([x, x_skip02])
    #x = l_act(x, act_type) 

    #------------------------------seq
    #x_skip02 = x
    #x_skip02 = attach_attention_module(x_skip02, 'se_block')

    #pooling
    if(is_pool):
        if pool_type == 'AP':
            x = l_avg_pool2d(x, pool_size=pool_size, pool_strides=pool_strides, pool_padding=pool_padding)
        elif pool_type == 'MP':
            x = l_max_pool2d(x, pool_size=pool_size, pool_strides=pool_strides, pool_padding=pool_padding)
        elif pool_type == 'GAP':
            x = l_glo_avg_pool2d(x)
        elif pool_type == 'G_MIX':
            x1 = l_glo_avg_pool2d(x)
            x2 = l_glo_max_pool2d(x)
            x  = x1 + x2
    #drop
    if is_drop:
        x = l_drop(x, drop_rate=drop_rate)

    return x
def res_inc01(x,
              f1, f2, f3, f4, 
              is_bn=True, bn_moment=0.9, bn_dim=-1,
              act_type='relu', 
              is_pool=True, pool_type='AP', pool_size=(2,2), pool_strides=(2,2), pool_padding='valid',
              is_drop=True, drop_rate=0.1
             ):
    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    x = l_conv2d(x, conv2d_filters=f1+f2+f3+f4, conv2d_kernel=(1,1))
    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    x = l_act(x, act_type)

    #skip
    x_skip = x

    #main
    x = l_conv2d(x, conv2d_filters=(f1+f2+f3+f4)/2, conv2d_kernel=(1,1))
    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    x = l_act(x, act_type) 

    x = l_inc01(x, f1, f2, f3, f4)
    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    x = l_act(x, act_type)

    x = l_inc01(x, f1, f2, f3, f4)
    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    x = l_act(x, act_type)

    x = l_conv2d(x, conv2d_filters=(f1+f2+f3+f4), conv2d_kernel=(1,1))

    #merge
    x = Add()([x, x_skip])
    x = l_act(x, act_type) 

    #pooling
    if(is_pool):
        if pool_type == 'AP':
            x = l_avg_pool2d(x, pool_size=pool_size, pool_strides=pool_strides, pool_padding=pool_padding)
        elif pool_type == 'MP':
            x = l_max_pool2d(x, pool_size=pool_size, pool_strides=pool_strides, pool_padding=pool_padding)
        elif pool_type == 'GAP':
            x = l_glo_avg_pool2d(x)
        elif pool_type == 'G_MIX':
            x1 = l_glo_avg_pool2d(x)
            x2 = l_glo_max_pool2d(x)
            x  = x1 + x2
    #drop
    if is_drop:
        x = l_drop(x, drop_rate=drop_rate)

    return x

def res_inc01_tb(x,
              f1, f2, f3, f4, 
              is_bn=True, bn_moment=0.9, bn_dim=-1,
              act_type='relu', 
              is_pool=True, pool_type='AP', pool_size=(2,2), pool_strides=(2,2), pool_padding='valid',
              is_drop=True, drop_rate=0.1
             ):

    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    x_skip = x
    x = l_conv2d(x, conv2d_filters=(f1+f2+f3+f4)/2, conv2d_kernel=(1,1))
    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    x = l_act(x, act_type) 

    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    x = l_inc01(x, f1, f2, f3, f4)
    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    x = l_act(x, act_type)

    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    x = l_inc01(x, f1, f2, f3, f4)
    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    x = l_act(x, act_type)

    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    x = l_conv2d(x, conv2d_filters=f1+f2+f3+f4, conv2d_kernel=(1,1))

    #skip
    x_skip = l_conv2d(x_skip, conv2d_filters=f1+f2+f3+f4, conv2d_kernel=(1,1))

    #merge
    x = Add()([x, x_skip])
    x = l_act(x, act_type) 

    #pooling
    if(is_pool):
        if pool_type == 'AP':
            x = l_avg_pool2d(x, pool_size=pool_size, pool_strides=pool_strides, pool_padding=pool_padding)
        elif pool_type == 'MP':
            x = l_max_pool2d(x, pool_size=pool_size, pool_strides=pool_strides, pool_padding=pool_padding)
        elif pool_type == 'GAP':
            x = l_glo_avg_pool2d(x)
        elif pool_type == 'G_MIX':
            x1 = l_glo_avg_pool2d(x)
            x2 = l_glo_max_pool2d(x)
            x  = x1 + x2
    #drop
    if is_drop:
        x = l_drop(x, drop_rate=drop_rate)

    return x

		
def l_res_01(x,
               conv2d_filters, conv2d_kernel, conv2d_padding='same', conv2d_dilation_rate=(1,1), 
               is_bn=True, bn_moment=0.9, bn_dim=-1,
               act_type='relu', 
               is_pool=True, pool_type='AP', pool_size=(2,2), pool_strides=(2,2), pool_padding='valid',
               is_drop=True, drop_rate=0.1
              ):

    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    x_skip = x    
    x = l_conv2d(x, 
                 conv2d_filters=conv2d_filters/2, 
                 conv2d_kernel=(1,1)
                )
    x = l_act(x, act_type)


    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    x = l_conv2d(x, 
                 conv2d_filters=conv2d_filters/2, 
                 conv2d_kernel=conv2d_kernel
                )
    x = l_act(x, act_type)

    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    x = l_conv2d(x, 
                 conv2d_filters=conv2d_filters, 
                 conv2d_kernel=(1,1)
                )
    #skip
    x_skip = l_conv2d(x_skip, 
                      conv2d_filters=conv2d_filters, 
                      conv2d_kernel=(1,1)
                     )

    #merge
    x = Add()([x, x_skip])
    x = l_act(x, act_type) 

    #pooling
    if(is_pool):
        if pool_type == 'AP':
            x = l_avg_pool2d(x, pool_size=pool_size, pool_strides=pool_strides, pool_padding=pool_padding)
        elif pool_type == 'MP':
            x = l_max_pool2d(x, pool_size=pool_size, pool_strides=pool_strides, pool_padding=pool_padding)
        elif pool_type == 'GAP':
            x = l_glo_avg_pool2d(x)
        elif pool_type == 'G_MIX':
            x1 = l_glo_avg_pool2d(x)
            x2 = l_glo_max_pool2d(x)
            x  = x1 + x2
    #drop
    if is_drop:
        x = l_drop(x, drop_rate=drop_rate)

    return x

def res_inc02(x,
              f1, f2, f3, f4, 
              is_bn=True, bn_moment=0.9, bn_dim=-1,
              act_type='relu', 
              is_pool=True, pool_type='AP', pool_size=(2,2), pool_strides=(2,2), pool_padding='valid',
              is_drop=True, drop_rate=0.1
             ):
    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    x_skip = x
    x = l_conv2d(x, conv2d_filters=(f1+f2+f3+f4)/2, conv2d_kernel=(1,1))
    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    x = l_act(x, act_type) 

    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    x = l_inc02(x, f1, f2, f3, f4)
    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    x = l_act(x, act_type)

    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    x = l_inc02(x, f1, f2, f3, f4)
    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    x = l_act(x, act_type)

    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    x = l_conv2d(x, conv2d_filters=f1+f2+f3+f4, conv2d_kernel=(1,1))

    #skip
    x_skip = l_conv2d(x_skip, conv2d_filters=f1+f2+f3+f4, conv2d_kernel=(1,1))

    #merge
    x = Add()([x, x_skip])
    x = l_act(x, act_type) 

    #pooling
    if(is_pool):
        if pool_type == 'AP':
            x = l_avg_pool2d(x, pool_size=pool_size, pool_strides=pool_strides, pool_padding=pool_padding)
        elif pool_type == 'MP':
            x = l_max_pool2d(x, pool_size=pool_size, pool_strides=pool_strides, pool_padding=pool_padding)
        elif pool_type == 'GAP':
            x = l_glo_avg_pool2d(x)
        elif pool_type == 'G_MIX':
            x1 = l_glo_avg_pool2d(x)
            x2 = l_glo_max_pool2d(x)
            x  = x1 + x2
    #drop
    if is_drop:
        x = l_drop(x, drop_rate=drop_rate)

    return x

def dob_inc02(x,
              f1, f2, f3, f4, 
              is_bn=True, bn_moment=0.9, bn_dim=-1,
              act_type='relu', 
              is_pool=True, pool_type='AP', pool_size=(2,2), pool_strides=(2,2), pool_padding='valid',
              is_drop=True, drop_rate=0.1
             ):

    #batchnorm 01
    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    #conv 01
    x = l_inc02(x, f1, f2, f3, f4)
    #act 01
    x = l_act(x, act_type)

    #batchnorm 02
    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    #conv 02    
    x = l_inc02(x, f1, f2, f3, f4)
    #act 02
    x = l_act(x, act_type)

    ##batchnorm 02

    #pooling
    if(is_pool):
        if pool_type == 'AP':
            x = l_avg_pool2d(x, pool_size=pool_size, pool_strides=pool_strides, pool_padding=pool_padding)
        elif pool_type == 'MP':
            x = l_max_pool2d(x, pool_size=pool_size, pool_strides=pool_strides, pool_padding=pool_padding)
        elif pool_type == 'GAP':
            x = l_glo_avg_pool2d(x)
        elif pool_type == 'G_MIX':
            x1 = l_glo_avg_pool2d(x)
            x2 = l_glo_max_pool2d(x)
            x  = x1 + x2
    #drop
    if is_drop:
        x = l_drop(x, drop_rate=drop_rate)

    return x

def l_dec_conv2d(x, pre_kernel_num, kernel_num, kernel_size):

    idex = int(pre_kernel_num/4)
    x33 = x[:,:,:,0:idex] +  x[:,:,:,idex:2*idex] +  x[:,:,:,2*idex:3*idex] +  x[:,:,:,3*idex:4*idex]

    conv_33 = Conv2D(kernel_num/4, kernel_size, padding='same', 
                   dilation_rate      = (1,1),
                   kernel_initializer = initializers.TruncatedNormal(mean=0.0,stddev=0.1),
                   kernel_regularizer = regularizers.l2(1e-5),
                   bias_initializer   = initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                   bias_regularizer   = regularizers.l2(1e-5)
                  )(x33)

    conv_11 = Conv2D(kernel_num/4, (1,1), padding='same',
                   dilation_rate      = (1,1),
                   kernel_initializer = initializers.TruncatedNormal(mean=0.0,stddev=0.1),
                   kernel_regularizer = regularizers.l2(1e-5),
                   bias_initializer   = initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                   bias_regularizer   = regularizers.l2(1e-5)
                  )(x)

    conv_11_h1 = Conv2D(kernel_num/4, (1,1), padding='same',
                   dilation_rate      = (1,1),
                   kernel_initializer = initializers.TruncatedNormal(mean=0.0,stddev=0.1),
                   kernel_regularizer = regularizers.l2(1e-5),
                   bias_initializer   = initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                   bias_regularizer   = regularizers.l2(1e-5)
                  )(x[:,:,:,0:int(pre_kernel_num/2)])

    conv_11_h2 = Conv2D(kernel_num/4, (1,1), padding='same',
                   dilation_rate      = (1,1),
                   kernel_initializer = initializers.TruncatedNormal(mean=0.0,stddev=0.1),
                   kernel_regularizer = regularizers.l2(1e-5),
                   bias_initializer   = initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                   bias_regularizer   = regularizers.l2(1e-5)
                   )(x[:,:,:,int(pre_kernel_num/2):])

    # concatenate filters, assumes filters/channels last
    x = Concatenate(axis=-1)([conv_33, conv_11_h1, conv_11_h2, conv_11])

    return x

def l_conv1d(x, conv1d_filters, conv1d_kernel, conv1d_strides=1, conv1d_padding='same', conv1d_dilation_rate=1):
    return  Conv1D(filters       = conv1d_filters,
                   kernel_size   = conv1d_kernel,
                   strides       = conv1d_strides,
                   padding       = conv1d_padding,
                   dilation_rate = conv1d_dilation_rate,

                   kernel_initializer=initializers.TruncatedNormal(mean=0.0,stddev=0.1),
                   kernel_regularizer=regularizers.l2(1e-5),
                   bias_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                   bias_regularizer=regularizers.l2(1e-5)
                  )(x)

