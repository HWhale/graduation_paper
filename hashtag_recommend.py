# -*- coding: utf-8 -*-
"""
Created on Wed May  1 11:33:00 2019

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

from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from scipy import spatial


def get_insta_image(id):
    img_data = requests.get('https://www.instagram.com/p/' + id + '/media/?size=m').content
    if not os.path.exists('./images'):
        os.makedirs('./images')
    with open('./images/' + id + '.jpg' , 'wb') as handler:
        handler.write(img_data)


# get image feature from pretrained model
base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)

def get_hash_data(name, dir_name, image_limit, post_limit):
    id_list = []
    img_list = []
    tag_list = []
    token_list = []
    combined_list = []
    
    #kkma = Kkma()
    filename = os.path.join(os.getcwd(), dir_name, name + '.csv')
    image_cnt = 0
    post_cnt = 0
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        
        # delete column name
        next(reader)
        
        # for each row, extract post_id and hashtags
        for line in reader:
            # if tag does not exist, skip current row
            tags = line[1].split('#')[1:]
            if len(tags) < 2:
                continue
            
            # extract id from row
            post_id = line[0].split('/')[4]
            
            # download and extract image from row
            # when the image count is less then image limit
            if image_cnt < image_limit:
                if not os.path.isfile('./images/'+post_id+'.jpg'):
                    get_insta_image(post_id)
                    print("donwload: " + post_id)
                try:
                    image = Image.open('./images/'+post_id+'.jpg').resize((224, 224))
                except:
                    print("Doesn't exist: " + post_id)
                    continue
                else:
                    image = np.array(image)
                    image = preprocess_input(image)
                    img_list.append(image)
                    image_cnt += 1
            
            
            # remove english tags
            i = 0
            while i < len(tags):
                if tp.is_english(tags[i]):
                    del tags[i]
                else:
                    i += 1
                    
            # extract tags and change it into training set
            tag_list.append(tags)
            tokens = tp.subst_split(tags)
            combined_tag = tp.split_combine_tag(tags, tokens)
            token_list.append(tokens)
            combined_list += combined_tag
        
            id_list.append(post_id)
            
            post_cnt += 1
            print(post_cnt)
            if post_cnt >= post_limit:
                break
            
    # get feature from given image list
    img_list = np.array(img_list)
    img_list = model.predict(img_list)
    
    return id_list, img_list, tag_list, token_list, combined_list


# list of the user dataset
training_file_list = [
        '먹스타그램',
        '미용',
        '반려동물',
        '셀카',
        '운동',
        '육아',
        '일상',
        '코디',
        '풍경',
        '휴가'
             ]

test_file_list = [
        'chagungwoo',
        'i_am_tofu'
        ]

image_limit = 300
post_limit = 1000

# make a training dataset
total_id_list, total_img_list, total_tag_list, total_token_list, total_combined_list = get_hash_data(training_file_list[0], 'data', image_limit, post_limit)
for training_file in training_file_list[1:]:
    id_list, img_list, tag_list, token_list, combined_list = get_hash_data(training_file, 'data', image_limit, post_limit)
    total_id_list += id_list
    total_img_list = np.append(total_img_list, img_list, axis=0)
    total_tag_list += tag_list
    total_token_list += token_list
    total_combined_list += combined_list
    
    
#total_feature_list = model.predict(total_img_list)

test_image_limit = 20
test_post_limit = 20

# make a test dataset
test_id_list, test_img_list, test_tag_list, test_token_list, test_combined_list = get_hash_data(test_file_list[0], 'data', image_limit, post_limit)
for test_file in test_file_list[1:]:
    id_list, img_list, tag_list, token_list, combined_list = get_hash_data(test_file, 'data', test_image_limit, test_post_limit)
    test_id_list += id_list
    test_img_list = np.append(test_img_list, img_list, axis=0)
    test_tag_list += tag_list
    test_token_list += token_list
    test_combined_list += combined_list

#test_feature_list = model.predict(test_img_list)

# length of total_img_list is gonna be a multiple of image_limit
# length of total_post_list is gonna be a multiple of post_limit
# by using it, you can find the post index which is corresponding to given image index
# image_index / image_limit * post_limit + image_index % image_limit

model_token = Word2Vec(total_token_list, window=100, min_count=50)

# make a probability function by using combined word
prob_dict = {}
for combined in total_combined_list:
    for i in range(len(combined) - 1):
        word_before = combined[i]
        word_after = combined[i+1]
        tp.make_next_prob(prob_dict, word_before, word_after)

# get nearest neighbor's hashtags
def nearest_neighbor_image(image, k):
    min_heap = []
    for i in range(k):
        heapq.heappush(min_heap, (1 - spatial.distance.cosine(image.flatten(), total_img_list[i].flatten()), i))
    
    for i in range(k, len(total_img_list)):
        heapq.heappushpop(min_heap, (1 - spatial.distance.cosine(image.flatten(), total_img_list[i].flatten()), i))
    
    return min_heap

# test input data which is a first entry of the feature list
for i in range(len(test_img_list)):
    neighbor_list = nearest_neighbor_image(test_img_list[i], 5)
    
    print('input post: ' + test_id_list[i])
    print(test_tag_list[i])
    
    neighbor_tokens = []
    for neighbor in neighbor_list:
        idx = neighbor[1]
        idx = idx // image_limit * post_limit + idx % image_limit
        print(total_id_list[idx])
        neighbor_tokens.append(total_token_list[idx])
        #print(total_token_list[neighbor_idx])
    
    main_words = tp.get_similar_words(model_token, neighbor_tokens, 0.99999)
    print(main_words)
    recommend_list = []
    for main_word in main_words:
        recommend_list += tp.extend_words(model_token, prob_dict, main_word, 0.999999, 0.40)
    recommend_list = list(set(recommend_list))
    print(recommend_list)
    
    #print('neighbor post: ' + total_id_list[idx])
    #print(total_post_list[idx])
