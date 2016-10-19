import numpy as np
import cPickle as pickle
import codecs
import matplotlib.pyplot as plt
plt.switch_backend('QT4Agg')
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

with open('flag_final.pkl','rb') as fp:
    flags=pickle.load(fp)

fname = [flag[0:50] for flag in flags]
flandmass = [flag[50:100] for flag in flags]
fhue = [flag[106:156] for flag in flags]
ftl = [flag[166:216] for flag in flags]
fbr = [flag[216:266] for flag in flags]

for i in range(2,11):
    model = TSNE(n_components=i, verbose=2, random_state=0)
    print "Reducing to "+str(i)+" dimensions"
    reduced_matrix = model.fit_transform(fname)
    np.save("../Flag_vectors/flag_6B_50_"+str(i)+"_name.npy",reduced_matrix)
    reduced_matrix = model.fit_transform(flandmass)
    np.save("../Flag_vectors/flag_6B_50_"+str(i)+"_land.npy",reduced_matrix)
    reduced_matrix = model.fit_transform(fhue)
    np.save("../Flag_vectors/flag_6B_50_"+str(i)+"_hue.npy",reduced_matrix)
    reduced_matrix = model.fit_transform(ftl)
    np.save("../Flag_vectors/flag_6B_50_"+str(i)+"_tl.npy",reduced_matrix)
    reduced_matrix = model.fit_transform(fbr)
    np.save("../Flag_vectors/flag_6B_50_"+str(i)+"_br.npy",reduced_matrix)
