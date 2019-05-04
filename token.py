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
                #print(len(cand[i]))
                for k in range(len(cand[i])):
                    if len(tag_list1[j].split(cand[i][k])) > 1 and len(cand[i][k]) > max_length:
                        max_length = len(cand[i][k])
                        max_string = cand[i][k]
        if max_length > 1:
            #print("i")
            #print(i)
            print(cand[i][k])
            try:
                token.index(max_string)
            except:
                token.append(max_string)
        elif max_length == 0:
            token.append(tag_list1[i])
    while True:
        origin = token
        token = exist_split(token)
        if origin == token:
            break
    print(tag_list1)
    print(token)

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
    #print(result)
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
    print(result1)
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
    
    print(result1)
    return result1

print(subst_split(['평창한우마을','강원도맛집','한우맛집','강원도한우맛집','홍천맛집','평창맛집','강원도맛집평창한우마을','어버이날선물','어린이날외식','감사선물']))
print(subst_split(['진솔할머니순두부','속초맛집','속초순두부맛집','속초여행','속초순두부','속초황태구이','속초맛집추천','설악산맛집','속초한정식','속초가볼만한곳','속초순두부마을','속초초당순두부','속초아침식사','속초순두부맛집진솔할머니순두부']))
print(subst_split(['동치미막국수','메밀전병','막국수맛집','먹스타그램','삼교리동치미막국수태장점','태장맛집','태장동맛집','원주맛집','원주태장동맛집','원주맛집삼교리동치미막국수','원주맛집','강원도막국수','막국수맛집','먹방그램','데일리','강원도맛집']))
print(subst_split(['등갈비달인','데이트맛집','분위기짱','매운맛집','직원들이친절해요','방송탄집','천호맛집','천호맛집등갈비달인최고','성내동맛집','강동구맛집','매운등갈비찜','등갈비맛집_등갈비달인','천호역맛집','등갈비최강맛집','천호등갈비달인','강동구맛집_등갈비달인','천호등갈비']))