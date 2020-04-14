#Looks for GPU

import tensorflow as tf
x=tf.test.is_gpu_available()
if x==False:
    print('GPU is down')
else:
    print('GPU is up')
