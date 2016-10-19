import numpy as np
import cPickle as pickle
import codecs
import matplotlib.pyplot as plt
plt.switch_backend('QT4Agg')
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

with open("../flag_dataset/flags_preprocessed.csv") as f:
    flag_ip = [line.rstrip() for line in f]

flag_label = [flag_ipi.split(",")[1] for flag_ipi in flag_ip]
# print flag_label
# remove first element i.e. title
flag_label = flag_label[1:]

with open('flag_final.pkl','rb') as fp:
    flags=pickle.load(fp)

fname = [flag[0:50] for flag in flags]
flandmass = [flag[50:100] for flag in flags]
ff = [flag[100:106] for flag in flags]
fhue = [flag[106:156] for flag in flags]
fs = [flag[156:166] for flag in flags]
ftl = [flag[166:216] for flag in flags]
fbr = [flag[216:266] for flag in flags]

# comment out after preparing vectors
# for i in range(2,11):
#     model = TSNE(n_components=i, verbose=2, random_state=0)
#     print "Reducing to "+str(i)+" dimensions"
#     reduced_matrix = model.fit_transform(fname)
#     np.save("../Flag_vectors/flag_6B_50_"+str(i)+"_name.npy",reduced_matrix)
#     reduced_matrix = model.fit_transform(flandmass)
#     np.save("../Flag_vectors/flag_6B_50_"+str(i)+"_land.npy",reduced_matrix)
#     reduced_matrix = model.fit_transform(fhue)
#     np.save("../Flag_vectors/flag_6B_50_"+str(i)+"_hue.npy",reduced_matrix)
#     reduced_matrix = model.fit_transform(ftl)
#     np.save("../Flag_vectors/flag_6B_50_"+str(i)+"_tl.npy",reduced_matrix)
#     reduced_matrix = model.fit_transform(fbr)
#     np.save("../Flag_vectors/flag_6B_50_"+str(i)+"_br.npy",reduced_matrix)

# fiddle with the number of clusters to get an optimal number
k_means = KMeans(n_clusters=2)

for i in range(2,11):
    comp = []
    fnames = np.load("../Flag_vectors/flag_6B_50_"+str(i)+"_name.npy")
    flands = np.load("../Flag_vectors/flag_6B_50_"+str(i)+"_land.npy")
    fhues = np.load("../Flag_vectors/flag_6B_50_"+str(i)+"_hue.npy")
    ftls = np.load("../Flag_vectors/flag_6B_50_"+str(i)+"_tl.npy")
    fbrs = np.load("../Flag_vectors/flag_6B_50_"+str(i)+"_br.npy")
    # print len(vectors)
    for j in range(0,len(flags)):
        tuplez = []
        tuplez.extend(fnames[j])
        tuplez.extend(flands[j])
        tuplez.extend(ff[j])
        tuplez.extend(fhues[j])
        tuplez.extend(fs[j])
        tuplez.extend(ftls[j])
        tuplez.extend(fbrs[j])
        # print tuplez
        comp.append(tuplez)

    tsne_model = TSNE(n_components=2, verbose=2, random_state=0)
    reduced_comp = tsne_model.fit_transform(comp)
    k_means.fit(reduced_comp)
    cluster_labels = k_means.labels_

    for i in set(k_means.labels_):
        index = k_means.labels_ == i
        plt.plot(reduced_comp[index,0], reduced_comp[index,1], 'o')
    for k in range(0,len(flag_label)):
        plt.annotate(flag_label[k],(reduced_comp[k,0],reduced_comp[k,1]))
    plt.show()
