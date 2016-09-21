import numpy as np
import cPickle as pickle
import codecs
from sklearn.manifold import TSNE

with open('glove.6B/Common4000/zoo_final.pkl','rb') as fp:
    names=pickle.load(fp)
animals = [name[0] for name in names]
# print animals
with open("glove.6B/glove.6B.50d.txt") as f:
    glove_ip = [line.rstrip() for line in f]
# print len(glove_ip)
glove_label = [glove_tmp.split(" ")[0] for glove_tmp in glove_ip]
glove_pre_emb = [glove_tmp.split(" ")[1:] for glove_tmp in glove_ip]

label_index = []
# toprint = ""
for name in animals:
    if name in glove_label:
        # toprint = toprint+name+" "+str(glove_label.index(name))+"\n"
        label_index.append(glove_label.index(name))
    else:
        # toprint = toprint+name+" -1\n"
        label_index.append(-1)
print label_index
# print toprint
processed = []
filtered = []
glove_emb = [map(float, glove_atmp) for glove_atmp in glove_pre_emb]
for j in range(0,len(label_index)):
    if label_index[j]!=-1:
        processed.append(glove_emb[label_index[j]])
        filtered.append(animals[j])
print filtered
input()
for i in range(2,11):
    model = TSNE(n_components=i, verbose=2, random_state=0)
    print "Reduced to "+str(i)+" dimensions"
    reduced_matrix = model.fit_transform(processed)
    np.save("zoo_6B_50_"+str(i)+"_vectors.npy",reduced_matrix)
