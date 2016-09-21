import numpy as np
import cPickle as pickle
# import matplotlib.pyplot as plt
# plt.switch_backend('QT4Agg')
import codecs
with open("zoo.data.txt") as f:
    zoo_ip = [line.rstrip() for line in f]
final = []
for zo in zoo_ip:
    dup = zo.split(",")
    # print dup
    final.append(dup)
zoo_final = []
for fin in final:
    zoo_temp = []
    for fi in range(0,len(fin)):
        if fi == 0:
            zoo_temp.append(fin[fi])
        else:
            zoo_temp.append(int(fin[fi]))
    zoo_final.append(zoo_temp)
# print zoo_final
with open('zoo_final.pkl','wb') as fp:
    pickle.dump(zoo_final,fp)
# with open('zoo_final.pkl','rb') as fp:
#     d1=pickle.load(fp)
# print d1
