# -*- coding: utf-8 -*-
"""
Created on Tue May  7 00:18:30 2019

@author: sungpil
"""

import csv
import os
import token_processor as tp


def refine(name_list, dir_name, min_freq, min_tag, max_tag):
    tag_dict = {}
    line_cnt = 0
    for name in name_list:
        filename = os.path.join(os.getcwd(), dir_name, name + '.csv')
        with open(filename, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            
            # delete column name
            next(reader)
            
            # for each row, extract post_id and hashtags
            for line in reader:
                # if tag does not exist, skip current row
                tags = line[1].split('#')[1:]
                for tag in tags:
                    tag = tp.get_only_korean(tag)
                    if tag == '':
                        continue
                    elif tag not in tag_dict:
                        tag_dict[tag] = 0
                    tag_dict[tag] += 1
                line_cnt += 1
    
    # after reading all files, print the number of total posts
    print("# of Total Posts: " + str(line_cnt))
          
    # drop tags less than min_freq
    for key in list(tag_dict.keys()):
        if tag_dict[key] < min_freq:
            del tag_dict[key]
    
    line_cnt = 0
    refined_name = os.path.join(os.getcwd(), dir_name, 'refined.csv')
    with open(refined_name, 'w', newline='', encoding='utf-8') as refined_csv:
        writer = csv.writer(refined_csv)
        writer.writerow(['id', 'hash', 'token', 'keyword'])
        for name in name_list:
            filename = os.path.join(os.getcwd(), dir_name, name + '.csv')
            with open(filename, 'r', encoding='utf-8') as file:
                
                # csv refinement by using new tags
                reader = csv.reader(file)
                for line in reader:
                    row = [line[0]]
                    tags = line[1].split('#')[1:]
                    tags_after = ''
                    tags_after_list = []
                    for tag in tags:
                        if tag in tag_dict:
                            tags_after += '#' + tag
                            tags_after_list.append(tag)
                    if min_tag <= len(tags_after_list) and len(tags_after_list) <= max_tag:
                        tokens = tp.subst_split(tags_after_list)
                        token_str = ''
                        for token in tokens:
                            token_str += '#' + token
                        row.append(tags_after)
                        row.append(token_str)
                        row.append(name)
                        writer.writerow(row)
                        
                        line_cnt += 1
    print("# of Refined Posts: " + str(line_cnt))
        
    return tag_dict

name_list = [
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

tag_dict = refine(name_list, 'data/unrefined', 30, 5, 20)