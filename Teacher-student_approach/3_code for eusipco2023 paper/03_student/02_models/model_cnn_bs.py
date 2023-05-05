
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Concatenate
#from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling1D
from tensorflow.keras.layers import Concatenate

import tensorflow as tf

from util import *

def model_cnn_bs(nF_aud, nT_aud, nC_aud, nClass, chanDim=-1):
   #----------------------------------------- input define
   input_shape_aud = (nF_aud, nT_aud, nC_aud)
   inputs_aud = Input(shape=input_shape_aud)

   #---------------------------------------- network architecture
   #---Audio stream
   #BLOCK 01:
   x= dob_conv2d(inputs_aud, 32, (3,3),
                 is_bn=True,
                 act_type='relu',
                 is_pool=False, pool_type='MP', pool_size=(2,2), pool_strides=(2,2),
                 drop_rate=0.1
                )
   x= dob_conv2d(x, 32, (3,3),
                 is_bn=True,
                 act_type='relu',
                 is_pool=True, pool_type='MP', pool_size=(2,2), pool_strides=(2,2),
                 drop_rate=0.1
                )
   #BLOCK 02: 
   x= dob_conv2d(x, 64, (3,3),
                 is_bn=True,
                 act_type='relu',
                 is_pool=False, pool_type='MP', pool_size=(2,2), pool_strides=(2,2),
                 drop_rate=0.1
                )
   x= dob_conv2d(x, 64, (3,3),
                 is_bn=True,
                 act_type='relu',
                 is_pool=True, pool_type='MP', pool_size=(2,2), pool_strides=(2,2),
                 drop_rate=0.1
                )

   #BLOCK 03:
   x= dob_conv2d(x, 128, (3,3),
                 is_bn=True,
                 act_type='relu',
                 is_pool=False, pool_type='MP', pool_size=(2,2), pool_strides=(2,2),
                 drop_rate=0.1
                )
   x= dob_conv2d(x, 128, (3,3),
                 is_bn=True,
                 act_type='relu',
                 is_pool=True, pool_type='MP', pool_size=(2,2), pool_strides=(2,2),
                 drop_rate=0.1
                )

   #BLOCK 04:
   x= dob_conv2d(x, 256, (3,3),
                 is_bn=True,
                 act_type='relu',
                 is_pool=False, pool_type='MP', pool_size=(2,2), pool_strides=(2,2),
                 drop_rate=0.1
                )
   x= dob_conv2d(x, 256, (3,3),
                 is_bn=True,
                 act_type='relu',
                 is_pool=True, pool_type='GAP', pool_size=(2,2), pool_strides=(2,2),
                 drop_rate=0.1
                )

   ###time
   #x1 = tf.math.reduce_mean(x, axis=1) 
   #x1 = tf.math.reduce_mean(x1, axis=2) 

   ##freq
   #x2 = tf.math.reduce_mean(x, axis=2) 
   #x2 = tf.math.reduce_mean(x2, axis=2) 

   ##channel
   #x3 = GlobalAveragePooling2D()(x)

   #print(np.shape(x1))
   #print(np.shape(x2))
   #print(np.shape(x3))
   #print(np.shape(x))

   #exit()

   #x1 = MultiHeadAttention(num_heads=16, key_dim=32)(x1, x1)
   #x2 = MultiHeadAttention(num_heads=16, key_dim=32)(x2, x2)

   #x1 = GlobalAveragePooling1D()(x1)
   #x2 = GlobalAveragePooling1D()(x2)

   #x1 = GlobalMaxPooling1D()(x1)
   #x2 = GlobalMaxPooling1D()(x2)

   #x  = Concatenate(axis=-1)([x1, x2])

   #---concatenate aud and img streams
   #BLOCK 07:
   y = l_dense(x, den_unit=4096,name_dense='output2')
   x = l_act(y, 'relu')
   x = l_drop(x, 0.2)

   #BLOCK 09:  
   x = l_dense(x, den_unit=45, name_dense='prob')
   x = l_act(x, 'softmax', "output1")

   #---output
   name='vgg'
   output = Model(inputs=inputs_aud, outputs=[x,y], name=name)

   return output
   

