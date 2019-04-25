# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 12:59:08 2019

@author: sungpil
"""

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from matplotlib import pyplot as plt
import numpy as np
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
import math

class Preprocessor_CIFAR10:
    @staticmethod
    def get():
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        return (x_train, y_train), (x_test, y_test)
    
    @staticmethod
    def get_partial_train(portion_list):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
       
        
        data_list = []
        for i in range(10):
            data_list.append([])
        for i in range(len(x_train)):
            data_list[y_train[i][0]].append(x_train[i])
            
        x_train = []
        y_train = []
        for i in range(len(portion_list)):
            index = int(len(data_list[i]) * portion_list[i])
            data_list[i] = data_list[i][:index]
            for j in range(index):
                x_train.append(data_list[i][j])
                y_train.append([i])
        
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)
        x_train = np.array(x_train, dtype='float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        
        perm = np.random.permutation(len(x_train))
        x_train = x_train[perm]
        y_train = y_train[perm]
                
        return (x_train, y_train), (x_test, y_test)
        
        
class SampleCNN(Sequential):
    def __init__(self, num_classes):
        Sequential.__init__(self)
        self.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)))
        self.add(BatchNormalization())
        self.add(Activation('relu'))
        self.add(Conv2D(32, (3, 3)))
        self.add(BatchNormalization())
        self.add(Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Dropout(0.25))
        self.add(Conv2D(64, (3, 3), padding='same'))
        self.add(BatchNormalization())
        self.add(Activation('relu'))
        self.add(Conv2D(64, (3, 3)))
        self.add(BatchNormalization())
        self.add(Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Dropout(0.25))
         
        self.add(Flatten())
        self.add(Dense(512))
        self.add(BatchNormalization())
        self.add(Activation('relu'))
        self.add(Dropout(0.5))
        self.add(Dense(num_classes))
        self.add(Activation('softmax'))
         
        self.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
(x_train, y_train), (x_test, y_test) = Preprocessor_CIFAR10.get()
model = SampleCNN(10)
hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), nb_epoch=30, batch_size=32, verbose=1)
scores = model.evaluate(x_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
 
#모델 시각
fig, loss_ax = plt.subplots()
 
acc_ax = loss_ax.twinx()
 
loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['acc'], 'b', label='train acc')
acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')
 
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax. set_ylabel('accuracy')
 
loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
 
plt.show()

# extract score vector from test
score_test = model.predict(x_test)
def score_to_wordset(score_vec_list, num_near, threshold):
    word_list = []
    near_list = []
    for score_vec in score_vec_list:
        center_label = np.argmax(score_vec)
        score_vec[center_label] = 0
        
        for i in range(num_near):
            if np.max(score_vec) < threshold:
                break
            word_list.append(center_label)
            near_label = np.argmax(score_vec)
            near_list.append([near_label])
            score_vec[near_label] = 0
    
    return word_list, near_list

def next_batch(word_list, near_list, data_idx, batch_cnt):
    batch_word = []
    batch_near = []
    data_idx = data_idx % len(word_list)
    for i in range(batch_cnt):
        batch_word.append(word_list[data_idx])
        batch_near.append(near_list[data_idx])
        data_idx = (data_idx + 1) % len(word_list)
    return batch_word, batch_near, data_idx
        
word_list, near_list = score_to_wordset(score_test, 2, 0.1)

class Word2Vec_Label_Model:
    def __init__(self, vocabulary_size, embedding_size, batch_size):
        # Step 4: skip-gram 모델 구축
        self.vocabulary_size = 10
        self.embedding_size = 2
        
        np.random.seed(1)
        tf.set_random_seed(1)
        
        self.batch_size = 128        # 일반적으로 16 <= batch_size <= 512
        
        #skip_window = 1         # target 양쪽의 단어 갯수
        #num_skips = 2           # 컨텍스트로부터 생성할 레이블 갯수
        
        num_sampled = 8    # negative 샘플링 갯수
        
        self.train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        self.train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        
        truncated = tf.truncated_normal([vocabulary_size, embedding_size],
                                        stddev=1.0 / math.sqrt(embedding_size))
        nce_weights = tf.Variable(truncated)
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
        
        # embeddings 벡터. embed는 바로 아래 있는 tf.nn.nce_loss 함수에서 단 1회 사용
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, self.train_inputs)
        
        # 배치 데이터에 대해 NCE loss 평균 계산
        nce_loss = tf.nn.nce_loss(weights=nce_weights,
                                  biases=nce_biases,
                                  labels=self.train_labels,
                                  inputs=embed,
                                  num_sampled=num_sampled,
                                  num_classes=vocabulary_size)
        self.loss = tf.reduce_mean(nce_loss)
        
        # SGD optimizer
        self.optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(self.loss)
        
        # 유사도를 계산하기 위한 모델. 학습 모델은 optimizer까지 구축한 걸로 종료.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        self.normalized_embeddings = embeddings / norm
        #valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        #similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
        
    def train(self, word_list, near_list, num_steps):
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
        
            average_loss, data_index = 0, 0
            for step in range(num_steps):
                batch_inputs, batch_labels, data_index = next_batch(word_list, near_list, data_index, self.batch_size)
        
                feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels}
                _, loss_val = session.run([self.optimizer, self.loss], feed_dict=feed_dict)
                average_loss += loss_val
        
                # 마지막 2000번에 대한 평균 loss 표시
                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    print('Average loss at step {} : {}'.format(step, average_loss))
                    average_loss = 0
            '''
                    if step % 10000 == 0:
                        sim = similarity.eval()         # (16, 50000)
            
                        for i in range(valid_size):
                            valid_word = ordered_words[valid_examples[i]]
            
                            top_k = 8
                            nearest = sim[i].argsort()[-top_k - 1:-1][::-1]
                            log_str = ', '.join([ordered_words[k] for k in nearest])
                            print('Nearest to {}: {}'.format(valid_word, log_str))
            '''
            self.embed_result = self.normalized_embeddings.eval()
            return self.embed_result
    
    def visualize(self, label_name):
        embed = self.embed_result
        embed = np.swapaxes(embed, 0, 1)
        for i in range(len(label_name)):
            plt.scatter(embed[0][i], embed[1][i])
            plt.annotate(label_name[i], xy=(embed[0][i], embed[1][i]), xytext=(5, 2), textcoords='offset points')
            
        

# Step 6: embeddings 시각화
label2Vec = Word2Vec_Label_Model(10, 2, 128)
embed = label2Vec.train(word_list, near_list, 10001)
label2Vec.visualize(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
    


'''
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'

    plt.figure(figsize=(18, 18))        # in inches

    for (x, y), label in zip(low_dim_embs, labels):
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)

try:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)

    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only])     # (500, 2)
    #labels = ordered_words[:plot_only]                                  # 재구성한 코드

    plot_with_labels(low_dim_embs, labels)

except ImportError:

    print('Please install sklearn, matplotlib, and scipy to show embeddings.')
'''