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
user_list = ['godjp']

def get_hash_data(name, dir_name):
    kkma = Kkma()
    filename = os.path.join(os.getcwd(), dir_name, name + '.csv')
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        
        # delete column name
        next(reader)
        
        # for each row, extract post_id and hashtags
        for line in reader:
            post_id = line[0].split('/')[4]
            print(post_id)
            if not os.path.isfile('./images/'+post_id+'.jpg'):
                get_insta_image(post_id)
                print("donwload: " + post_id)
            #print(post_id)
            tags = line[1].split('#')[1:]
            for tag in tags:
                words = kkma.pos(tag, flatten=False)
                #print(tag, words)
                if len(words) > 1:
                    pass
                    #print(tag, words)    

for user in user_list:
    get_hash_data(user, 'data')



# set image to fixed size
image = Image.open('./images/BscipNvFcqi.jpg').resize((320, 320))
image = np.array(image)
image = preprocess_input(image)

# get image feature from pretrained model
model = VGG19(weights='imagenet', include_top=False)
feature = model.predict(image.reshape((1, 320, 320, 3)))





model = api.load('glove-twitter-50')
vec_woman = np.array(model['woman'])
vec_woman -= 0.1
model.wv.most_similar(positive=[vec_woman], topn=3)

kkma = Kkma()
print(kkma.pos('절대신뢰'))


