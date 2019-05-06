# -*- coding: utf-8 -*-
"""
Created on Fri May  3 20:14:54 2019

@author: sungpil
"""

import heapq

tags = ['평창한우마을', '강원도맛집', '한우맛집', '강원도한우맛집', '홍천맛집', '평창맛집', '강원도맛집평창한우마을', '어버이날선물', '어린이날외식', '감사선물']
tokens = ['마을', '강원도', '한우', '맛집', '홍천', '평창', '어버이날', '어린이날외식', '감사', '선물']

def is_english(string):
    for char in string:
        if ('a' <= char and char <= 'z') or ('A' <= char and char <= 'Z'):
            return True
    return False


def make_next_prob(prob_dict, word_before, word_after):
    if is_english(word_before) or is_english(word_after):
        return
    
    if word_before not in prob_dict:
        prob_dict[word_before] = {'count':0, 'next':{}}
        
    prob_dict[word_before]['count'] += 1
    if word_after not in prob_dict[word_before]['next']:
        prob_dict[word_before]['next'][word_after] = 1
    else:
        prob_dict[word_before]['next'][word_after] += 1

def drop_minimum(prob_dict, min_cnt):
    for word_before, sub_dict in prob_dict.items():
        for word_after, count_dict in prob_dict[word_before]['next'].items():
            count = prob_dict[word_before]['next'][word_after]
            if count < min_cnt:
                del prob_dict[word_before]['next'][word_after]
                prob_dict[word_before]['count'] -= count
        if prob_dict[word_before]['count'] < min_cnt:
            del prob_dict[word_before]

def get_next_prob(prob_dict, word, min_cnt):
    
    if word not in prob_dict:
        return []
    
    count = prob_dict[word]['count']
    if count < min_cnt:
        return []
    
    next_dict = prob_dict[word]['next']
    prob_list = []
    
    for next_word in next_dict:
        prob = next_dict[next_word] / count
        prob_list.append((next_word, prob))
    return prob_list

def get_similar_words(model_token, post_list, threshold_prob):
    similar_set = set([])
    
    for token_list in post_list:
        i = 0
        while i < len(token_list):
            if token_list[i] not in model_token.wv.vocab:
                del token_list[i]
            else:
                i += 1
    
    for i in range(len(post_list)):
        main_list = post_list[i]
        other_list = []
        for j in range(len(post_list)):
            if i != j:
                other_list += post_list[j]
        
        for word1 in main_list:
            for word2 in other_list:
                sim = model_token.wv.similarity(word1, word2)
                if sim >= threshold_prob:
                    similar_set.add(word1)
                    similar_set.add(word2)
                
    return list(similar_set)

def extend_words(model_token, prob_dict, start_word, word_threshold, next_threshold):
    pool = [('', start_word, 1)]
    res = []
    while pool:
        word = pool.pop()
        prev_word = word[0]
        curr_word = word[1]
        curr_prob = word[2]
        res.append(prev_word + curr_word)
        if curr_word not in model_token.wv.vocab:
            continue
        # transition to similar word
        for similar_word in model_token.wv.most_similar(curr_word):
            probability = similar_word[1] * curr_prob
            if probability >= word_threshold:
                pool.append((prev_word, similar_word[0], probability * 0.95))
        # extend to next word
        for next_word in get_next_prob(prob_dict, curr_word, 50):
            probability = next_word[1] * curr_prob
            if probability >= next_threshold and (prev_word+curr_word).find(next_word[0]) == -1:
                pool.append((prev_word + curr_word, next_word[0], curr_prob * 0.95))
    return res
    

def exist_split(tag_list):
    #tokenize with existing tag
    list_cnt = len(tag_list)
    temp1 = tag_list[:]
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