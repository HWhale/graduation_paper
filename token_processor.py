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
            
    
#print(split_combine_tag(tags, tokens))