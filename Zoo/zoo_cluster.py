import numpy as np
import cPickle as pickle
import codecs
import matplotlib.pyplot as plt
plt.switch_backend('QT4Agg')
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

with open('glove.6B/Common4000/zoo_final.pkl','rb') as fp:
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
    comp = []
    vectors = np.load("Zoo_Vectors/zoo_6B_50_"+str(i)+"_vectors.npy")
    # print len(vectors)
    for j in range(0,len(names)):
        tuplez = []
        tuplez.extend(vectors[j])
        tuplez.extend(names[j][1:])
        # print tuplez
        comp.append(tuplez)
    # print comp
    tsne_model = TSNE(n_components=2, verbose=2, random_state=0)
    reduced_comp = tsne_model.fit_transform(comp)
    k_means.fit(reduced_comp)
    cluster_labels = k_means.labels_
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
    # input()
    for i in set(k_means.labels_):
        index = k_means.labels_ == i
        plt.plot(reduced_comp[index,0], reduced_comp[index,1], 'o')
    for k in range(0,len(animals)):
        plt.annotate(animals[k],(reduced_comp[k,0],reduced_comp[k,1]))
    plt.show()
