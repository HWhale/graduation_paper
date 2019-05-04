# -*- coding: utf-8 -*-
"""
Created on Wed May  1 11:33:00 2019

@author: sungpil
"""

import gensim.downloader as api
#from konlpy.tag import Kkma
from gensim.models import Word2Vec
import numpy as np
import requests
import os
import csv
from matplotlib import pyplot as plt
from PIL import Image
import keras
from keras.models import Model

from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from scipy import spatial


def get_insta_image(id):
    img_data = requests.get('https://www.instagram.com/p/' + id + '/media/?size=m').content
    if not os.path.exists('./images'):
        os.makedirs('./images')
    with open('./images/' + id + '.jpg' , 'wb') as handler:
        handler.write(img_data)




def flatten_words(words):
    result = []
    #print(words)
    for big_word in words:
        for word in big_word:
            if word[1] == 'NNG' or word[1] == 'NNP':
                result.append(word[0])
    return result

# get image feature from pretrained model
base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)

def get_hash_data(name, dir_name, image_limit, post_limit):
    id_list = []
    img_list = []
    post_list = []
    
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
            
            # extract tags and change it into training set
            post_sentence = []
            for tag in tags:
                post_sentence.append(tag)
                #words = kkma.pos(tag, flatten=False)
                #print(tag, words)
                #if len(words) > 1:
                #   post_sentence += flatten_words(words)
            post_list.append(post_sentence)
        
            id_list.append(post_id)
            
            post_cnt += 1
            print(post_cnt)
            if post_cnt >= post_limit:
                break
            
    # get feature from given image list
    img_list = np.array(img_list)
    img_list = model.predict(img_list)
    
    return id_list, img_list, post_list


# list of the user dataset
training_file_list = [
        '¸Ô½ºÅ¸±×·¥',
        '¹Ì¿ë',
        '¹Ý·Áµ¿¹°',
        '¼¿Ä«',
        '¿îµ¿',
        'À°¾Æ',
        'ÀÏ»ó',
        'ÄÚµð',
        'Ç³°æ',
        'ÈÞ°¡'
             ]

test_file_list = [
        'chagungwoo'
        ]

# get nearest neighbor's hashtags
def nearest_neighbor_image(image):
    min_index = 0
    max_sim = 1 - spatial.distance.cosine(image.flatten(), total_img_list[0].flatten())
    for i in range(1, len(total_img_list)):
        sim = 1 - spatial.distance.cosine(image.flatten(), total_img_list[i].flatten())
        if sim > max_sim:
            min_index = i
            max_sim = sim
    return min_index

def exist_split(tag_list):
    #tokenize with existing tag
    list_cnt = len(tag_list)
    temp1 = tag_list
    for j in range(list_cnt):
        try:
            token = temp1[j]
        except:
            break
        for k in range(list_cnt):
            if k != j:
                try:
                    split_token = temp1[k].split(token)
                except:
                    break
                if len(split_token) > 1:
                    del temp1[k]
                    insert_idx = 0
                    for m in range(len(split_token)):
                        if split_token[m] != '' and len(split_token[m]) > 1:
                            temp1.insert(k + insert_idx, split_token[m])
                            insert_idx += 1
                if list_cnt != temp1:
                    list_cnt = len(temp1)
    list_cnt = len(temp1)
    
    return temp1

def subst_split(tag_list1):
    cand = []
    token = []
    result = []
    max_length = 0
    max_string = ''
    #print("tag")
    #print(tag_list1)
    for i in range(len(tag_list1)):
        cand.append([])
        for j in range(2, len(tag_list1[i]) + 1):
            for k in range(len(tag_list1[i]) - j + 1):
                try:
                    for m in range(j):
                        temp = tag_list1[i][k * j + m : k * j + j + m]
                        if temp != '' and len(temp) > 1:
                            try:
                                cand[i].index(temp)
                            except:
                                cand[i].append(temp)                                
                except:
                    break
            
    for i in range(len(tag_list1)):
        max_length = 0
        max_string = ''
        for j in range(len(tag_list1)):
            if j != i :
                for k in range(len(cand[i])):
                    if len(tag_list1[j].split(cand[i][k])) > 1 and len(cand[i][k]) > max_length:
                        max_length = len(cand[i][k])
                        max_string = cand[i][k]
        if max_length > 1:
            try:
                token.index(max_string)
            except:
                token.append(max_string)
        elif max_length == 0:
            token.append(tag_list1[i])
    #print(token)
    

    #while True:
    #    origin = token
    #    token = exist_split(token)
    #    if origin == token:
    #        break
    for i in range(len(tag_list1)):
        for j in range(len(token)):
            temp = tag_list1[i].split(token[j])
            if len(temp) > 1:
                for k in range(len(temp)):
                    if temp[k] != '' and len(temp[k]) > 1:
                        try:
                            result.index(temp[k])
                        except:
                            result.append(temp[k])
                    elif temp[k] == '' and len(token[j]) > 1:
                        try:
                            result.index(token[j])
                        except:
                            result.append(token[j])
            else:
                result.append(token[j])
    result += token
    result = list(set(result))
    result1 = token
    while True:
        origin = result1
        temp = exist_split(result)
        for i in range(len(temp)):
            for j in range(len(result)):
                temp2 = result[j].split(temp[i])
                for k in range(len(temp2)):
                    if len(temp2[k]) > 1:
                        result1.append(temp2[k])
                    else:
                        result1.append(temp[i])
        if len(result1) == len(origin):
            break
    result1 = list(set(result1))
    while True:
        origin = result1
        for i in range(len(result1)):
            for j in range(len(result1)):
                if i != j:
                    try:
                        temp = result1[j].split(result1[i])
                        if len(temp) > 1:
                            del result1[j]
                            for k in range(len(temp)):
                                if len(temp[k]) > 1:
                                   result1.append(temp[k])
                    except:
                        continue
        if origin == result1:
            break
    result1 = list(set(result1))
    return result1

def split_combine_tag(tags, tokens):
    combined_tags = []
    for tag in tags:
        token_list = []
        while tag != '':
            found = 0
            for token in tokens:
                if tag.find(token) == 0:
                    token_list.append(token)
                    tag = tag[len(token):]
                    found = 1
                    break
            if found == 0:
                tag = tag[1:]

        if len(token_list) >= 2:
            combined_tags.append(token_list)
    return combined_tags

image_limit = 300
post_limit = 300

# make a training dataset
total_id_list, total_img_list, total_post_list = get_hash_data(training_file_list[0], 'data', image_limit, post_limit)
total_wtow_list = []
for training_file in training_file_list[1:]:
    id_list, img_list, post_list = get_hash_data(training_file, 'data', image_limit, post_limit)
    total_id_list += id_list
    total_img_list = np.append(total_img_list, img_list, axis=0)
    total_post_list += post_list
for training_file in training_file_list[1:]:
    id_list, img_list, post_list = get_hash_data(training_file, 'data', image_limit, post_limit)
    total_id_list += id_list
    total_img_list = np.append(total_img_list, img_list, axis=0)
    total_post_list += post_list
    for i in range(len(post_list)):
        temp = split_combine_tag(post_list[i],subst_split(post_list[i]))
        print(temp)
        temp_list = []
        for j in range(len(temp)):
            temp_list += temp[j]
        total_wtow_list.append(temp_list)

print(total_wtow_list)
    
    
#total_feature_list = model.predict(total_img_list)

test_image_limit = 20
test_post_limit = 20

# make a test dataset
test_id_list, test_img_list, test_post_list = get_hash_data(test_file_list[0], 'data', image_limit, post_limit)
for test_file in test_file_list[1:]:
    id_list, img_list, post_list = get_hash_data(test_file, 'data', test_image_limit, test_post_limit)
    test_id_list += id_list
    test_img_list = np.append(test_img_list, img_list, axis=0)
    test_post_list += post_list

#test_feature_list = model.predict(test_img_list)

# length of total_img_list is gonna be a multiple of image_limit
# length of total_post_list is gonna be a multiple of post_limit
# by using it, you can find the post index which is corresponding to given image index
# image_index / image_limit * post_limit + image_index % image_limit

model_tag_to_tag = Word2Vec(total_post_list)
model_word_to_word = Word2Vec(total_wtow_list)


# test input data which is a first entry of the feature list
for i in range(len(test_img_list)):
    print(i)
    idx = nearest_neighbor_image(test_img_list[i])
    idx = idx // image_limit * post_limit + idx % image_limit
    print('input post: ' + test_id_list[i])
    print(test_post_list[i])
    #print("after tokenizing with substring")
    #print(subst_split(test_post_list[i]))
    #print("after tokenizing with existing tag")
    #print(exist_split(test_post_list[i]))
    print("split with token")
    cand = split_combine_tag(test_post_list[i],subst_split(test_post_list[i]))
    print(cand)
    print("similar words")
    for j in range(len(cand)):
        for k in range(len(cand[j])):
            print(cand[j][k])
            try:
                print("tagtotag")
                print(model_tag_to_tag.wv.most_similar(cand[j][k]))
                print("wordtoword")
                print(model_word_to_word.wv.most_similar(cand[j][k]))
                
            except:
                print("doesn't exist")
    print('neighbor post: ' + total_id_list[idx])
    print(total_post_list[idx])
    print(i)