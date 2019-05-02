# -*- coding: utf-8 -*-
"""
Created on Wed May  1 11:33:00 2019

@author: sungpil
"""

import gensim.downloader as api
from konlpy.tag import Kkma
from konlpy.utils import pprint
from gensim.models import Word2Vec
import numpy as np
import requests
import os
import csv
from matplotlib import pyplot as plt
from PIL import Image
import keras

from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input


def get_insta_image(id):
    img_data = requests.get('https://www.instagram.com/p/' + id + '/media/?size=m').content
    if not os.path.exists('./images'):
        os.makedirs('./images')
    with open('./images/' + id + '.jpg' , 'wb') as handler:
        handler.write(img_data)


# list of the user dataset
user_list = [
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

def flatten_words(words):
    result = []
    print(words)
    for big_word in words:
        for word in big_word:
            if word[1] == 'NNG' or word[1] == 'NNP':
                result.append(word[0])
    return result

def get_hash_data(name, dir_name, image_limit):
    id_list = []
    img_list = []
    post_list = []
    
    kkma = Kkma()
    filename = os.path.join(os.getcwd(), dir_name, name + '.csv')
    image_cnt = 0
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        
        # delete column name
        next(reader)
        
        # for each row, extract post_id and hashtags
        for line in reader:
            # extract id from row
            post_id = line[0].split('/')[4]
            id_list.append(post_id)
            
            # download and extract image from row
            # when the image count is less then image limit
            if image_cnt < image_limit:
                if not os.path.isfile('./images/'+post_id+'.jpg'):
                    get_insta_image(post_id)
                    print("donwload: " + post_id)
                image = Image.open('./images/'+post_id+'.jpg').resize((320, 320))
                image = np.array(image)
                image = preprocess_input(image)
                img_list.append(image)
                image_cnt += 1
            
            # extract tags and change it into training set
            tags = line[1].split('#')[1:]
            post_sentence = []
            for tag in tags:
                words = kkma.pos(tag, flatten=False)
                #print(tag, words)
                if len(words) > 1:
                    post_sentence += flatten_words(words)
            post_list.append(post_sentence)
    return id_list, np.array(img_list), post_list

for user in user_list:
    id_list, img_list, training_data = get_hash_data(user, 'data', 500)

model_word = Word2Vec(training_data, size=100, min_count=5, iter=100)
model_word.wv['제주']

# get image feature from pretrained model
model = VGG19(weights='imagenet', include_top=False)
feature_list = model.predict(img_list)

# get nearest neighbor's hashtags
def nearest_neighbor_image(image):
    min_index = 0
    min_dist = np.linalg.norm(image - feature_list[0])
    for i in range(1, len(feature_list)):
        dist = np.linalg.norm(image - feature_list[i])
        if dist < min_dist:
            min_index = i
            min_dist = dist
    return training_data[min_index]

# test input data which is a first entry of the feature list
input_feature = feature_list[0]
nearest_neighbor_image(input_feature)
        