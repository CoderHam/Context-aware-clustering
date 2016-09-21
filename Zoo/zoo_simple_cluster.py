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
labels = [name[0] for name in names]
animals = [name[1:] for name in names]
tsne_model = TSNE(n_components=2, verbose=2, random_state=0)
reduced_comp = tsne_model.fit_transform(animals)
k_means = KMeans(n_clusters=7)
k_means.fit(reduced_comp)
for i in set(k_means.labels_):
    index = k_means.labels_ == i
    plt.plot(reduced_comp[index,0], reduced_comp[index,1], 'o')
for k in range(0,len(animals)):
    plt.annotate(labels[k],(reduced_comp[k,0],reduced_comp[k,1]))
plt.show()
