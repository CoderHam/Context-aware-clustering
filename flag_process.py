import numpy as np
import codecs
with open("flag_dataset/flags_preprocessed.csv") as f:
    flag_ip = [line.rstrip() for line in f]
zone = ['NE','NW','SE','SW']
flag_list = [flag_ipi.split(",")[1:] for flag_ipi in flag_ip]
flag_labels = flag_list[0]
# print flag_labels
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

print final[0]

with open("glove.6B/glove.6B.50d.txt") as f:
    glove_ip = [line.rstrip() for line in f]
# print len(glove_ip)
glove_label = [glove_tmp.split(" ")[0] for glove_tmp in glove_ip]
glove_pre_emb = [glove_tmp.split(" ")[1:] for glove_tmp in glove_ip]

not_found_0 = []
not_found_1 = []
not_found_5 = []
for elem in final:
    if elem[0] not in glove_label:
        not_found_0.append(elem[0])
    if elem[1] not in glove_label:
        not_found_1.append(elem[1])
    if elem[5] not in glove_label:
        not_found_5.append(elem[5])

print not_found_0
#What about the first one??
#['american-samoa', 'antigua-barbuda', 'british-virgin-isles', 'cape-verde-islands', 'cayman-islands', 'central-african-republic', 'comorro-islands', 'cook-islands', 'costa-rica', 'dominican-republic', 'el-salvador', 'equatorial-guinea', 'falklands-malvinas', 'french-guiana', 'french-polynesia', 'germany-ddr', 'germany-frg', 'ivory-coast', 'maldive-islands', 'netherlands-antilles', 'new-zealand', 'north-korea', 'north-yemen', 'papua-new-guinea', 'puerto-rico', 'san-marino', 'sao-tome', 'saudi-arabia', 'sierra-leone', 'soloman-islands', 'south-korea', 'south-yemen', 'sri-lanka', 'st-helena', 'st-kitts-nevis', 'st-lucia', 'st-vincent', 'trinidad-tobago', 'turks-cocos-islands', 'us-virgin-isles', 'vatican-city', 'western-samoa']
print not_found_1
#For North/South America, Average N/S and America
#['n.america', 's.america']
print not_found_5
#Need a solution for ['other indo-european', 'japanese/turkish/finnish/magyar']
