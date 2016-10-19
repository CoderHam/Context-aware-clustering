import numpy as np
import codecs
import cPickle as pickle
with open("flag_dataset/flags_preprocessed.csv") as f:
    flag_ip = [line.rstrip() for line in f]
# zone is represented by a number - 0,1,2,3
zone = ['NE','NW','SE','SW']
flag_list = [flag_ipi.split(",")[1:] for flag_ipi in flag_ip]
#ignore columns from 10-16 because of missing values
flag_labels = flag_list[0][:10] + flag_list[0][17:]
print flag_labels
ints = [3,4,7,8,9,18,19,20,21,22,23,24,25,26,27]
final = []
for elemi in flag_list[1:]:
    temp = []
    for j in range(0,len(elemi)):
        if j in ints:
            temp.append(int(elemi[j]))
        elif j==2:
            temp.append(zone.index(elemi[j]))
        else:
            temp.append(elemi[j].lower())
    final.append(temp)

#extract and check for labels in GloVE
with open("glove.6B/glove.6B.50d.txt") as f:
    glove_ip = [line.rstrip() for line in f]
glove_label = [glove_tmp.split(" ")[0] for glove_tmp in glove_ip]
glove_pre_emb = [glove_tmp.split(" ")[1:] for glove_tmp in glove_ip]
glove_emb = [map(float, glove_atmp) for glove_atmp in glove_pre_emb]

#ignore columns from 10-16 because of missing values
for i in range(0,len(final)):
    final[i] = final[i][:10] + final[i][17:]
print final[0]
not_found_0 = []
not_found_1 = []
not_found_5 = []
not_found_6 = []
#check for elements not in glove
flag_list_final = []
sumtemp = [0] * 50
contemp = [0] * 50
for elem in final:
    flag_final = []
    if elem[0] not in glove_label:
        # not_found_0.append(elem[0])
        locelem = elem[0].split("-")
        for loce in locelem:
            if loce in glove_label:
                temp1 = glove_emb[glove_label.index(loce)]
                for i in range(0,50):
                    sumtemp[i] = temp1[i] + sumtemp[i]
                #add arrays
        avtemp = [x/len(locelem) for x in sumtemp]
        flag_final.extend(sumtemp)
    else:
        flag_final.extend(glove_emb[glove_label.index(elem[0])])
        # print glove_emb[glove_label.index(elem[0])]
    if elem[1] not in glove_label:
        # not_found_1.append(elem[1])
        if elem[1]=="n.america":
            direct = glove_emb[glove_label.index("north")]
        else:
            direct = glove_emb[glove_label.index("south")]
        amer = glove_emb[glove_label.index("america")]
        for i in range(0,50):
            contemp[i] = direct[i] + amer[i]
        #add arrays
        avtemp = [x/2 for x in contemp]
        flag_final.extend(contemp)
    else:
        flag_final.extend(glove_emb[glove_label.index(elem[1])])
    flag_final.append(elem[2])
    flag_final.append(elem[3])
    flag_final.append(elem[4])
    flag_final.append(elem[7])
    flag_final.append(elem[8])
    flag_final.append(elem[9])
    if elem[10] in glove_label:
        flag_final.extend(glove_emb[glove_label.index(elem[10])])
    else:
        print elem[10] + " not found"
    flag_final.append(elem[11])
    flag_final.append(elem[12])
    flag_final.append(elem[13])
    flag_final.append(elem[14])
    flag_final.append(elem[15])
    flag_final.append(elem[16])
    flag_final.append(elem[17])
    flag_final.append(elem[18])
    flag_final.append(elem[19])
    flag_final.append(elem[20])
    if elem[21] in glove_label:
        flag_final.extend(glove_emb[glove_label.index(elem[21])])
    else:
        print elem[21] + " not found"
    if elem[22] in glove_label:
        flag_final.extend(glove_emb[glove_label.index(elem[22])])
    else:
        print elem[22] + " not found"
    flag_list_final.append(flag_final)

with open('flag_final.pkl','wb') as fp:
    pickle.dump(flag_list_final,fp)
print len(flag_list_final)
#print unique not found in glove list for respective columns
print list(set(not_found_0))
#What about the multi-word locations? ['british-virgin-isles', 'st-lucia', 'papua-new-guinea', 'sierra-leone', 'st-vincent', 'saudi-arabia', 'vatican-city', 'san-marino', 'germany-frg', 'north-yemen', 'cook-islands', 'north-korea', 'el-salvador', 'falklands-malvinas', 'puerto-rico', 'french-polynesia', 'us-virgin-isles', 'sao-tome', 'equatorial-guinea', 'comorro-islands', 'germany-ddr', 'maldive-islands', 'dominican-republic', 'st-helena', 'netherlands-antilles', 'ivory-coast', 'costa-rica', 'french-guiana', 'antigua-barbuda', 'central-african-republic', 'new-zealand', 'turks-cocos-islands', 'american-samoa', 'sri-lanka', 'south-korea', 'south-yemen', 'st-kitts-nevis', 'trinidad-tobago', 'cayman-islands', 'western-samoa', 'cape-verde-islands', 'soloman-islands']
print list(set(not_found_1))
#For North/South America, Average N/S with America ['n.america', 's.america']
# print list(set(not_found_5))
#Need a solution for ['other indo-european', 'japanese/turkish/finnish/magyar']
# print list(set(not_found_6))
#Need a solution for ['other christian']
