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

# list of the user dataset
user_list = ['kotteunji']

def get_hash_data(name, dir_name):
    kkma = Kkma()
    filename = os.path.join(os.getcwd(), dir_name, name + '.csv')
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        # column name 제거
        for line in reader:
            tags = line[1].split('#')[1:]
            for tag in tags:
                words = kkma.pos(tag, flatten=False)
                #print(tag, words)
                if len(words) > 1:
                    print(tag, words)    

get_hash_data('kotteunji', 'data')


def get_insta_image(id):
    img_data = requests.get('https://www.instagram.com/p/' + id + '/media/?size=m').content
    if not os.path.exists('./images'):
        os.makedirs('./images')
    with open('./images/' + id + '.jpg' , 'wb') as handler:
        handler.write(img_data)
        
get_insta_image('Bbq2RxWjkTL')


model = api.load('glove-twitter-50')
vec_woman = np.array(model['woman'])
vec_woman -= 0.1
model.wv.most_similar(positive=[vec_woman], topn=3)

kkma = Kkma()
print(kkma.pos('절대신뢰'))


