import numpy as np
import codecs
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

#ignore columns from 10-16 because of missing values
for i in range(0,len(final)):
    final[i] = final[i][:10] + final[i][17:]
print final[0]
not_found_0 = []
not_found_1 = []
not_found_5 = []
not_found_6 = []
#check for elements not in glove
for elem in final:
    if elem[0] not in glove_label:
        not_found_0.append(elem[0])
    if elem[1] not in glove_label:
        not_found_1.append(elem[1])
    if elem[5] not in glove_label:
        not_found_5.append(elem[5])
    if elem[6] not in glove_label:
        not_found_6.append(elem[6])
#print unique not found in glove list for respective columns
print list(set(not_found_0))
#What about the multi-word locations? ['british-virgin-isles', 'st-lucia', 'papua-new-guinea', 'sierra-leone', 'st-vincent', 'saudi-arabia', 'vatican-city', 'san-marino', 'germany-frg', 'north-yemen', 'cook-islands', 'north-korea', 'el-salvador', 'falklands-malvinas', 'puerto-rico', 'french-polynesia', 'us-virgin-isles', 'sao-tome', 'equatorial-guinea', 'comorro-islands', 'germany-ddr', 'maldive-islands', 'dominican-republic', 'st-helena', 'netherlands-antilles', 'ivory-coast', 'costa-rica', 'french-guiana', 'antigua-barbuda', 'central-african-republic', 'new-zealand', 'turks-cocos-islands', 'american-samoa', 'sri-lanka', 'south-korea', 'south-yemen', 'st-kitts-nevis', 'trinidad-tobago', 'cayman-islands', 'western-samoa', 'cape-verde-islands', 'soloman-islands']
print list(set(not_found_1))
#For North/South America, Average N/S with America ['n.america', 's.america']
print list(set(not_found_5))
#Need a solution for ['other indo-european', 'japanese/turkish/finnish/magyar']
print list(set(not_found_6))
#Need a solution for ['other christian']
