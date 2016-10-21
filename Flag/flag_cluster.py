import numpy as np
import cPickle as pickle
import codecs
import matplotlib.pyplot as plt
plt.switch_backend('QT4Agg')
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

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
k_means = KMeans(n_clusters=4)

#Elbow Test
def elbow_test(comp,n_dim):
    inertia = []
    for i in xrange(2,15,1):
        model = KMeans(n_clusters=i)
        model.fit(comp)
        inertia.append(model.inertia_)
    plt.plot(range(2,15,1),inertia)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.savefig('../Flag_elbow_test/elbow_test_'+str(n_dim)+'.png')
    plt.clf()

# Silhouette Analysis
def silhouette_analysis(comp):
    for n_clusters in xrange(2,15,1):
        X = np.array(comp)
        fig, ax1 = plt.subplots(1)
        ax1.set_xlim([-1, 1])
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        print "For n_clusters =", n_clusters,"The average silhouette_score is :", silhouette_avg
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
            0, ith_cluster_silhouette_values,
            facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhoutte score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        plt.savefig('../Flag_Sillhoutte_Analysis/clusters_'+str(n_clusters)+'.png', bbox_inches='tight')


for i in range(2,11):
    print "Dimensionality = "+str(i)
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

    # Elbow test
    # elbow_test(comp,i)

    # Silhouette Analysis
    # if i==10:
    #     silhouette_analysis(comp)

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
