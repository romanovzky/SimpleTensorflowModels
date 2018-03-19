import tensorflow as tf
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np

data=load_breast_cancer()
#data=load_iris()

scaler = StandardScaler()
lb = LabelBinarizer()

X_data=(scaler.fit_transform(data.data)).astype(np.float32)
Y_cls=data.target
Y_data = lb.fit_transform(Y_cls).astype(np.int64)

X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.3,random_state=30)

n_features = X_train.shape[1]
n_classes=Y_train.shape[1]

learning_rate=0.01

def neuron_layer(X,n_neurons,name,activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.shape[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs,n_neurons),stddev=stddev)
        W=tf.Variable(init,name="Weights")
        b=tf.Variable(tf.zeros([n_neurons]),name="biases")
        z=tf.matmul(X,W)+b
        if activation=="relu":
            return tf.nn.relu(z)
        if activation=="leaky_relu":
            return tf.nn.leaky_relu(z)
        if activation=="sigmoid":
            return tf.nn.sigmoid(z)
        if activation=="softmax":
            return tf.nn.softmax(z)
        else:
            return z

tf.reset_default_graph()

n_hidden_1 = 8
n_hidden_2 = 8

X = tf.placeholder(tf.float32,shape=[None,n_features],name="X")
y = tf.placeholder(tf.int64,shape=[None,n_classes],name="y")

with tf.name_scope("DNN"):
    hidden1 = neuron_layer(X,n_hidden_1,"hidden1",activation="leaky_relu")
    hidden2 = neuron_layer(hidden1,n_hidden_2,"hidden2",activation="leaky_relu")
    logits = neuron_layer(hidden2,n_classes,"outputs")
with tf.name_scope("loss"):
     loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=logits),name="avg_xentropy")
with tf.name_scope("train"):    
    optimizer =  tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op= optimizer.minimize(loss)
with tf.name_scope("accuracy"): 
    correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
with tf.name_scope("init"):
    init=tf.global_variables_initializer()

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

n_epochs = 100
batch_size = 50

with tf.Session() as sess:
    init.run()
    max_acc=0
    acc_going_down=0
    for epoch in range(n_epochs):
        batch_step=0
        avg_loss = 0.
        total_loss= 0.
        total_batch = int(X_train.shape[0]/batch_size)
        for X_batch, Y_batch in iterate_minibatches(X_train,Y_train,batchsize=batch_size):
            _,l=sess.run([training_op,loss],feed_dict={X:X_batch, y:Y_batch})
            batch_step+=1
            total_loss += l
        if((epoch)%10==0):
            avg_loss = total_loss/batch_size
            print("Epoch:", '%02d' % (epoch+1), "| Average Training Loss= {:.2f}".format(avg_loss), "| Training Accuracy:  {:.2f}".format(accuracy.eval({X: X_train, y: Y_train})), "| Test Accuracy:  {:.2f}".format(accuracy.eval({X: X_test, y: Y_test})))
    print("Model fit complete.")
    print("Final Training Accuracy: {:.2f}".format(accuracy.eval({X: X_train, y: Y_train})))
    print("Final Validation Accuracy: {:.2f}".format(accuracy.eval({X: X_test, y: Y_test})))
