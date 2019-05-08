# -*- coding: utf-8 -*-
"""
Created on Tue May  7 13:08:11 2019

@author: sungpil
"""

from gensim.models import Word2Vec
import numpy as np
import requests
import os
import csv
from matplotlib import pyplot as plt
from PIL import Image
from keras.models import Model
import heapq
import token_processor as tp
import random

from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from scipy import spatial


def get_insta_image(id):
    img_data = requests.get('https://www.instagram.com/p/' + id + '/media/?size=m').content
    if not os.path.exists('./images'):
        os.makedirs('./images')
    with open('./images/' + id + '.jpg' , 'wb') as handler:
        handler.write(img_data)

def load_image(address):
    addr_id = address.split('/')[4]
    if not os.path.isfile('./images/'+addr_id+'.jpg'):
        return None
    
    image = Image.open('./images/'+addr_id+'.jpg').resize((200, 200))
    
    return image


# get image feature from pretrained model
base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)

def image_extract(total_post, image_limit):
    image_post = []
    image_data = []
    for post in total_post:
        if len(image_post) >= image_limit:
            break
        
        # extract id from row
        post_id = post[0].split('/')[4]
        print(post_id)
        if not os.path.isfile('./images/'+post_id+'.jpg'):
            get_insta_image(post_id)
            print("donwload: " + post_id)
        try:
            image = Image.open('./images/'+post_id+'.jpg').resize((224, 224))
        except Exception as e:
            print(e)
            continue
        else:
            image = np.array(image)
            image = preprocess_input(image)
            image_data.append(image)
            image_post.append(post)
            
    return image_post, image_data


def build_dataset(name, dir_name, training_post_limit, training_image_limit, test_limit):
    filename = os.path.join(os.getcwd(), dir_name, name + '.csv')
    row_list = []
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        
        # delete column name
        next(reader)
        
        for line in reader:
            row = []
            row.append(line[0])
            row.append(line[1].split('#')[1:])
            row.append(line[2].split('#')[1:])
            row_list.append(row)
            
    print('shuffle..')
    random.shuffle(row_list)
    print('shuffle done')
    
    # training_post must include training_image
    # test_limit must exclude training_post
    training_post = row_list[:training_post_limit]
    training_image_post, training_image_data = image_extract(training_post, training_image_limit)
    test_post, test_image_data = image_extract(row_list[training_post_limit:], test_limit)
    del row_list
    
    batch_size = 400
    batch_cnt = training_image_limit // batch_size
    for batch_i in range(batch_cnt):
        batch_image_data = np.array(training_image_data[batch_i*batch_size:(batch_i+1)*batch_size])
    
        print('get feature map from training images.. batch ' + str(batch_i))
        batch_image_data = model.predict(batch_image_data)
        # append feature map to training_image_post
        for i in range(batch_size):
            training_image_post[batch_i*batch_size + i].append(batch_image_data[i])
    
    test_image_data = np.array(test_image_data)
    print('get feature map from test images..')
    test_image_data = model.predict(test_image_data)
    # append feature map to test_image_post
    for i in range(len(test_post)):
        test_post[i].append(test_image_data[i])
    
    # for training_post, only hash tag are needed
    training_token = []
    training_combined = []
    for post in training_post:
        tag = post[1]
        token = post[2]
        combined = tp.split_combine_tag(tag, token)
        training_token.append(token)
        training_combined += combined
    
    # preprocessing for each dataset
    print(len(training_image_data), len(test_image_data))
    
    
    
    
    
    # training_token        : list of tokens in each post (2D List)
    # training_combined     : list of combined hashtags in a form of a list of tokens
    # training_image_post   : training posts used in KNN search with feature map
    # test_post             : test posts
    
    # form of post
    # 0: string, instagram address
    # 1: list, splitted original tag
    # 2: list, splitted token
    # 3: nparray, feature map
    
    return training_token, training_combined, training_image_post, test_post
    
    
# build training and test datset
training_token, training_combined, training_image_post, test_post = build_dataset('refined', 'data', 20000, 4000, 1000)

# train word2vec by using token_list
model_token = Word2Vec(training_token, window=100, min_count=10)

# make a probability function by using combined word
prob_dict = {}
for combined in training_combined:
    for i in range(len(combined) - 1):
        word_before = combined[i]
        word_after = combined[i+1]
        tp.make_next_prob(prob_dict, word_before, word_after)


# get nearest neighbor's hashtags
def nearest_neighbor_image(post_input, dataset_post, k):
    post_input = post_input[3].flatten()
    min_heap = []
    
    for post in dataset_post[:k]:
        heapq.heappush(min_heap, (1 - spatial.distance.cosine(post_input, post[3].flatten()), post))
        
    for post in dataset_post[k:]:
        heapq.heappushpop(min_heap, (1 - spatial.distance.cosine(post_input, post[3].flatten()), post))
        
    return min_heap


k = 5
correct_min = 1
total_correct_num = [0, 0, 0]
for post in test_post: 
    neighbor_list = nearest_neighbor_image(post, training_image_post, k)
    print('input post: ' + post[0])
    
    fig = plt.figure(figsize=((1+k)*2, 7))
    #plt.subplot(1, k+1, 1)
    
    #plt.axis('off')
    #plt.imshow(load_image(post[0]))
    
    neighbor_num = 1
    neighbor_tokens = []
    for neighbor in neighbor_list:
        neighbor = neighbor[1]
        neighbor_tokens.append(neighbor[2])
        #plt.subplot(1, k+1, neighbor_num+1)
        #plt.axis('off')
        #plt.imshow(load_image(neighbor[0]))
        neighbor_num += 1
    #fig.tight_layout()
    #fig.show()
    
    main_words = tp.get_similar_words(model_token, neighbor_tokens, 0.98)
    
    print(post[1])
    print(main_words)
    recommend_list = []
    for main_word in main_words:
        recommend_list += tp.extend_words(model_token, prob_dict, main_word, 0.999, 0.40)
        
    recommend_list = list(set(recommend_list))
    print(recommend_list)
    
    correct_num = 0
    for answer in post[1]:
        for recommend in recommend_list:
            if recommend.find(answer) != -1:
                #print(answer, recommend)
                correct_num += 1
    for i in range(3):
        if correct_num >= i+1:
            total_correct_num[i] += 1
    
for i in range(3):
    print('Performance result: ' + str(total_correct_num[i]/len(test_post)))
