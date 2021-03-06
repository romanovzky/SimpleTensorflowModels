import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")

height = 28
width = 28
n_inputs = height * width
channels = 1
input_shape = (height, width, channels)
batch_size = 50
num_classes = 10
epochs = 10

X_train = mnist.train.images
X_val = mnist.validation.images
X_test = mnist.test.images

X_train_reshaped = X_train.reshape(X_train.shape[0], height, width, channels)
X_val_reshaped = X_val.reshape(X_val.shape[0], height, width, channels)
X_test_reshaped = X_test.reshape(X_test.shape[0], height, height, channels)


Y_train = mnist.train.labels
Y_val = mnist.validation.labels
Y_test = mnist.test.labels

conv1_fmaps = 64
conv1_ksize = 3
conv1_stride = 1
conv1_pad = "SAME"
conv1_act=tf.nn.relu

conv2_fmaps = 64
conv2_ksize = 3
conv2_stride = 1
conv2_pad = "SAME"
conv2_act=tf.nn.relu

pool1_fmaps = conv2_fmaps
pol1_ksize=[1, 2, 2, 1]
pol1_strides=[1, 2, 2, 1]
pol1_padding="VALID"

dropout_1 = 0.25

conv3_fmaps = 32
conv3_ksize = 3
conv3_stride = 1
conv3_pad = "SAME"
conv3_act=tf.nn.relu

conv4_fmaps = 32
conv4_ksize = 3
conv4_stride = 1
conv4_pad = "SAME"
conv4_act=tf.nn.relu

pool2_fmaps = conv4_fmaps
pol2_ksize=[1, 2, 2, 1]
pol2_strides=[1, 2, 2, 1]
pol2_padding="VALID"


dropout_2 = 0.25

dense1_neurons = 256
dense1_act=tf.nn.relu

dropout_3 = 0.5

n_outputs = num_classes

tf.reset_default_graph()

with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32,shape=[None,n_inputs],name="X")
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
    y = tf.placeholder(tf.int64,shape=(None),name="y")
    training = tf.placeholder_with_default(False, shape=[], name='training')
    
with tf.name_scope("ConvPolLayer1"):
    convulution_1 = tf.layers.conv2d(X_reshaped,
                                 filters=conv1_fmaps,
                                 strides=conv1_stride,
                                 kernel_size=conv1_ksize,
                                 padding=conv1_pad,
                                 activation=conv1_act,
                                 name="conv1")
    convulution_2 = tf.layers.conv2d(convulution_1,
                                 filters=conv2_fmaps,
                                 strides=conv2_stride,
                                 kernel_size=conv2_ksize,
                                 padding=conv2_pad,
                                 activation=conv2_act,
                                 name="conv2")
    pool_1 = tf.nn.avg_pool(convulution_2,
                            ksize=pol1_ksize,
                            strides=pol1_strides,
                            padding=pol1_padding,
                            name="pol1")
    pool_1_drop = tf.layers.dropout(pool_1,
                                   dropout_1,
                                   training=training)
    
with tf.name_scope("ConvPolLayer2"):
    convulution_3 = tf.layers.conv2d(pool_1_drop,
                                 filters=conv3_fmaps,
                                 strides=conv3_stride,
                                 kernel_size=conv3_ksize,
                                 padding=conv3_pad,
                                 activation=conv3_act,
                                 name="conv3")
    convulution_4 = tf.layers.conv2d(convulution_3,
                                 filters=conv4_fmaps,
                                 strides=conv4_stride,
                                 kernel_size=conv4_ksize,
                                 padding=conv4_pad,
                                 activation=conv4_act,
                                 name="conv4")
    pool_2 = tf.nn.avg_pool(convulution_4,
                            ksize=pol2_ksize,
                            strides=pol2_strides,
                            padding=pol2_padding,
                            name="pol1")
    pool_2_drop = tf.layers.dropout(pool_2,
                                   dropout_2,
                                   training=training)    
    
with tf.name_scope("Flatten"):
    shape = pool_2_drop.get_shape().as_list()
    last_flat = tf.reshape(pool_2_drop,[-1,shape[1]*shape[2]*shape[3]])
with tf.name_scope("Dense1"):
    dense1 = tf.layers.dense(last_flat,
                             units=dense1_neurons,
                             activation=dense1_act,
                             name="fc1")
    dense1_drop = tf.layers.dropout(dense1,
                                    dropout_3,
                                    training=training)
with tf.name_scope("Output"):
    logits = tf.layers.dense(dense1_drop,
                             n_outputs,
                             name="output")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")
    
with tf.name_scope("Train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)    
with tf.name_scope("Eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
with tf.name_scope("Init"):
    init = tf.global_variables_initializer()

n_epochs = 10
batch_size = 50
n_batches = mnist.train.num_examples // batch_size

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
      for iteration in range(n_batches):
        X_batch, Y_batch = mnist.train.next_batch(batch_size)
        batch_loss, batch_acc, _ = sess.run([loss,accuracy,training_op], feed_dict={X: X_batch, y: Y_batch,training:True})            
      epoch_loss_val = loss.eval(feed_dict={X: X_val,y: Y_val})
      epoch_acc_val = accuracy.eval(feed_dict={X: X_val,y: Y_val})        
      print("-"*50)
      print("Epoch {} | Last Batch Train Loss: {:.4f} | Last Batch Train Accuracy: {:.4f} | Validation Loss: {:.4f} | Validation Accuracy: {:.4f} ".format(epoch+1, batch_loss , batch_acc, epoch_loss_val , epoch_acc_val ))
    acc_test = accuracy.eval(feed_dict={X: X_test,y: Y_test})
    print("Final accuracy on test set:", acc_test)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, AveragePooling2D
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from keras import backend as K

K.clear_session()

model = Sequential()

model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape,padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), activation='relu',padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu',padding='same'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer="Adam",
              metrics=['accuracy'])

print(model.summary())
model.fit(X_train_reshaped, Y_train, batch_size=batch_size, epochs=epochs,validation_data=(X_val_reshaped, Y_val))
test_pred=model.predict_classes(X_test_reshaped)
print("Final accuracy on test set:", accuracy_score(test_pred,Y_test))

