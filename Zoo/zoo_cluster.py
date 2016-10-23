import numpy as np
import cPickle as pickle
import codecs
import matplotlib.pyplot as plt
plt.switch_backend('QT4Agg')
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

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
    plt.savefig('../Zoo_elbow_test/elbow_test_'+str(n_dim)+'.png')
    plt.clf()

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

        plt.savefig('../Zoo_Sillhoutte_Analysis/clusters_'+str(n_clusters)+'.png', bbox_inches='tight')

with open('../glove.6B/Common4000/zoo_final.pkl','rb') as fp:
    names=pickle.load(fp)
filtered = ['aardvark', 'antelope', 'bass', 'bear', 'boar', 'buffalo', 'calf', 'carp', 'catfish', 'cavy', 'cheetah', 'chicken', 'chub', 'clam', 'crab', 'crayfish', 'crow', 'deer', 'dogfish', 'dolphin', 'dove', 'duck', 'elephant', 'flamingo', 'flea', 'frog', 'frog', 'giraffe', 'girl', 'gnat', 'goat', 'gorilla', 'gull', 'haddock', 'hamster', 'hare', 'hawk', 'herring', 'honeybee', 'housefly', 'kiwi', 'ladybird', 'lark', 'leopard', 'lion', 'lobster', 'lynx', 'mink', 'mole', 'mongoose', 'moth', 'newt', 'octopus', 'opossum', 'oryx', 'ostrich', 'parakeet', 'penguin', 'pheasant', 'pike', 'piranha', 'pitviper', 'platypus', 'polecat', 'pony', 'porpoise', 'puma', 'pussycat', 'raccoon', 'reindeer', 'rhea', 'scorpion', 'seahorse', 'seal', 'sealion', 'skimmer', 'skua', 'slug', 'sole', 'sparrow', 'squirrel', 'starfish', 'stingray', 'swan', 'termite', 'toad', 'tortoise', 'tuatara', 'tuna', 'vampire', 'vole', 'vulture', 'wallaby', 'wasp', 'wolf', 'worm', 'wren']
# print len(filtered)
toremove = []
for name in names:
    if name[0] not in filtered:
        toremove.append(name)
for rem in toremove:
    names.remove(rem)
# print len(names)
category = [name[-1] for name in names]
animals = [name[0] for name in names]
k_means = KMeans(n_clusters=7)
for i in range(2,11):
    print "Dimensionality = "+str(i)
    comp = []
    vectors = np.load("../Zoo_Vectors/zoo_6B_50_"+str(i)+"_vectors.npy")
    # print len(vectors)
    for j in range(0,len(names)):
        tuplez = []
        tuplez.extend(vectors[j])
        tuplez.extend(names[j][1:])
        # print tuplez
        comp.append(tuplez)

    # Elbow test
    # elbow_test(comp,i)

    # Silhouette Analysis
    if i==10:
        silhouette_analysis(comp)

    # tsne_model = TSNE(n_components=2, verbose=2, random_state=0)
    # reduced_comp = tsne_model.fit_transform(comp)
    # k_means.fit(reduced_comp)
    # cluster_labels = k_means.labels_
    # print len(cluster_labels)
    # print len(category)
    # for k in range(0,len(animals)):
    #     plt.annotate(animals[k],(reduced_comp[k,0],reduced_comp[k,1]))
    # plt.show()
    # input()
    # for ani in range(0,len(category)):
    #     if cluster_labels[ani] == category[ani]:
    #         print str(animals[ani]) +" correctly belongs to "+ str(category[ani])
    #     else:
    #         print str(animals[ani]) +" incorrectly belongs to "+ str(category[ani])
    # for i in set(k_means.labels_):
    #     index = k_means.labels_ == i
    #     plt.plot(reduced_comp[index,0], reduced_comp[index,1], 'o')
    # for k in range(0,len(animals)):
    #     plt.annotate(animals[k],(reduced_comp[k,0],reduced_comp[k,1]))
    # plt.show()
