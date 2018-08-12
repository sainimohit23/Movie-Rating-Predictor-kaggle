import pandas as pd
from sentiment_utils import *
import tensorflow as tf
import keras.backend as k
import numpy as np
import re


train_data = pd.read_table('train.tsv')
X_train = train_data.iloc[:,2]
Y_train = train_data.iloc[:,3]


from sklearn.preprocessing import OneHotEncoder
Y_train = Y_train.reshape(Y_train.shape[0],1)
ohe = OneHotEncoder(categorical_features=[0])
Y_train = ohe.fit_transform(Y_train).toarray()


maxLen = len(max(X_train, key=len).split())
m = X_train.shape[0]



words_to_index, index_to_words, word_to_vec_map = read_glove_vectors("glove/glove.6B.50d.txt")
emb_matrix = np.zeros((len(words_to_index)+1, word_to_vec_map['go'].shape[0]))


for i in range(1, len(words_to_index)):
    emb_matrix[i] = word_to_vec_map[str(index_to_words[i])]


strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


def sentance_to_indices(X_train, words_to_index, maxLen):
    m = X_train.shape[0]
    X_indices = np.zeros((m, maxLen))
    
    for i in range(m):
        cleaned_sentance = cleanSentences(X_train[i])
        sentance_words = cleaned_sentance.lower().strip().split()
        
        j = 0
        for word in sentance_words:
            try:
                X_indices[i, j] = words_to_index[word]
            except KeyError:
                X_indices[i, j] = words_to_index['unk']
            j += 1
    
    return X_indices


X_train_indices = sentance_to_indices(X_train, words_to_index, maxLen)    
    
batchSize = 24
lstmUnits = 64
numClasses = 5
iterations = 500001
num_layers = 2

def getTrainBatch():
    num = np.random.randint(0, X_train.shape[0]-25)
    X_batch = X_train_indices[num:num+batchSize,:]
    Y_batch = Y_train[num:num+batchSize, :]
    
    return X_batch, Y_batch






tf.reset_default_graph()
inputs = tf.placeholder(tf.int32, [batchSize, maxLen])
labels = tf.placeholder(tf.float32, [batchSize, numClasses])


data = tf.Variable(tf.zeros([batchSize, maxLen, 50]),dtype=tf.float32)
data = tf.nn.embedding_lookup(emb_matrix,inputs)



cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(lstmUnits), 0.5) for _ in range(num_layers)])    
value, _ =  tf.nn.dynamic_rnn(cells,data,dtype=tf.float64)

weight = tf.get_variable('weights', shape=[lstmUnits, numClasses], initializer= tf.contrib.layers.xavier_initializer(), dtype= tf.float64)
bias = tf.get_variable('bias', shape=[numClasses], initializer=tf.zeros_initializer(), dtype= tf.float64)
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)


correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)


# COPIED
import datetime

sess = tf.InteractiveSession()
tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)


#sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('models/'))

for i in range(iterations):
   #Next Batch of reviews
   #print('iteration no : ' + str(i))
   nextBatch, nextBatchLabels = getTrainBatch();
   _, curloss = sess.run([optimizer, loss], {inputs: nextBatch, labels: nextBatchLabels}) 
   #Write summary to Tensorboard
   if (i % 50 == 0):
       print("Loss at iteration " + str(i) + " : " + str(curloss))
       summary = sess.run(merged, {inputs: nextBatch, labels: nextBatchLabels})
       writer.add_summary(summary, i)
       
   if (i%10000 == 0):
       saver.save(sess, 'models/pretrained_model.ckpt', global_step=i)
       
writer.close()
































