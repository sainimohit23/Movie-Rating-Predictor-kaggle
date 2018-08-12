import pandas as pd
from sentiment_utils import *
import keras
import keras.backend as k
import numpy as np


train_data = pd.read_table('train.tsv')
X_train = train_data.iloc[:,2]
Y_train = train_data.iloc[:,3]


from sklearn.preprocessing import OneHotEncoder
Y_train = Y_train.reshape(Y_train.shape[0],1)
ohe = OneHotEncoder(categorical_features=[0])
Y_train = ohe.fit_transform(Y_train).toarray()


maxLen = len(max(X_train, key=len).split())
words_to_index, index_to_words, word_to_vec_map = read_glove_vectors("glove/glove.6B.50d.txt")
m = X_train.shape[0]



def sentance_to_indices(X_train, words_to_index, maxLen, dash_index_list, keys):
    m = X_train.shape[0]    
    X_indices = np.zeros((m, maxLen))
    
    for i in range(m):
        if i in dash_index_list:
            continue
        
        sentance_words = X_train[i].lower().strip().split()
        
        j = 0
        for word in sentance_words:
            if word in keys:
                X_indices[i, j] = words_to_index[word]
            j += 1
    
    return X_indices






def pretrained_embedding_layer(word_to_vec_map, words_to_index):
    emb_dim = word_to_vec_map['pen'].shape[0]
    vocab_size = len(words_to_index) + 1
    emb_matrix = np.zeros((vocab_size, emb_dim))
    
    for word, index in words_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]
    
    emb_layer= keras.layers.embeddings.Embedding(vocab_size, emb_dim, trainable= False)
    
    emb_layer.build((None,))
    emb_layer.set_weights([emb_matrix])
    
    return emb_layer



def get_model(input_shape, word_to_vec_map, words_to_index):
    
    sentance_indices = keras.layers.Input(shape = input_shape, dtype='int32')
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, words_to_index)
    embeddings = embedding_layer(sentance_indices)
    
    X = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True))(embeddings)
    X = keras.layers.Dropout(0.5)(X)
    
    X = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=False))(X)
    X = keras.layers.Dropout(0.5)(X)
    X = keras.layers.Dense(5)(X)
    
    X = keras.layers.Activation('softmax')(X)
    
    model = keras.models.Model(sentance_indices, X)
    
    return model



model = get_model((maxLen,), word_to_vec_map, words_to_index)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

dash_index_list = []
for i in range(m):
    if '-' in X_train[i]:
        dash_index_list.append(i)


# either use below line to process data or use pre-processed data provided as "X_train_indices"
X_train_indices = sentance_to_indices(X_train, words_to_index, maxLen, dash_index_list, keys)


X_train_indices = pd.read_csv('X_train_indices.csv')
X_train_indices = X_train_indices.iloc[:, 1:]


model.fit(X_train_indices, Y_train, epochs = 30, batch_size = 32, shuffle=True)
model.save('model.h5')





"""                         Testing Model                    """
keys = []
for key in word_to_vec_map.keys():
     keys.append(key)


X_test = pd.Series(["Movie was very good"])
X_test_indices = sentance_to_indices(X_test, words_to_index, maxLen, [], keys)
pred = model.predict(X_test_indices)
















