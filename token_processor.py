# -*- coding: utf-8 -*-
"""
Created on Fri May  3 20:14:54 2019

@author: sungpil
"""

#tags = ['평창한우마을', '강원도맛집', '한우맛집', '강원도한우맛집', '홍천맛집', '평창맛집', '강원도맛집평창한우마을', '어버이날선물', '어린이날외식', '감사선물']
#tokens = ['마을', '강원도', '한우', '맛집', '홍천', '평창', '어버이날', '어린이날외식', '감사', '선물']

def split_combine_tag(tags, tokens):
    combined_tags = []
    for tag in tags:
        token_list = []
        while tag != '':
            for token in tokens:
                if tag.find(token) == 0:
                    token_list.append(token)
                    tag = tag[len(token):]
                    break
        if len(token_list) >= 2:
            combined_tags.append(token_list)
    return combined_tags

def exist_split(test_post_list):
    #tokenize with existing tag
    list_cnt = len(test_post_list)
    for j in range(list_cnt):
        try:
            token = test_post_list[j]
        except:
            break
        for k in range(j + 1, list_cnt):
            try:
                split_token = test_post_list[k].split(token)
            except:
                break
            if len(split_token) > 1:
                del test_post_list[k]
                insert_idx = 0
                for m in range(len(split_token)):
                    if split_token[m] != '' and len(split_token[m]) > 1:
                        test_post_list.insert(k + insert_idx, split_token[m])
                        insert_idx += 1
            if list_cnt != test_post_list:
                list_cnt = len(test_post_list)
    list_cnt = len(test_post_list)
    for j in range(list_cnt - 1, 0, -1):   
        for k in range(list_cnt - 2, j, -1):
            try:
                split_token = test_post_list[k].split(token)
            except:
                break
            if len(split_token) > 1:
                del test_post_list[k]
                insert_idx = 0
                for m in range(len(split_token)):
                    if split_token[m] != '' and len(split_token[m]) > 1:
                        test_post_list.insert(k + insert_idx, split_token[m])
                        insert_idx += 1
            if list_cnt != test_post_list:
                list_cnt = len(test_post_list)
    return test_post_list
    
#print(split_combine_tag(tags, tokens))