import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('QT4Agg')
import codecs
from sklearn.manifold import TSNE
#
with open("glove.6B.50d.txt") as f:
    glove_ip = [line.rstrip() for line in f]
# print len(glove_ip)
glove_label = [glove_tmp.split(" ")[0] for glove_tmp in glove_ip]
glove_pre_emb = [glove_tmp.split(" ")[1:] for glove_tmp in glove_ip]
#
#
# #for glove_temp in glove_pre_emb:
# #    print glove_temp #convert to np and append to np_emb
# #    a= input()
#
glove_emb = [map(float, glove_atmp) for glove_atmp in glove_pre_emb]
#
# # np_emb = np.asarray(glove_emb)
# # np_label = np.asarray(glove_label)
# # np.save("emb_6B_300_all.npy",np_emb)
# # np.save("label_6B_200_all.npy",np_label)
#
# print len(glove_emb)
# # glove_matrix = np.loadtxt("glove.6B.300d.embeddings.txt")
# # glove_words = [line.strip() for line in open("glove.6B.300d.labels.txt")]
#
target_words = [line.strip().lower() for line in open("4000-most-common-english-words-csv.csv")][:2000]
#
rows = [glove_label.index(word) for word in target_words if word in glove_label]
target_matrix = [glove_emb[row] for row in rows]
#reduced_matrix = tsne(target_matrix, 2)
# target_matrix = np.load("Common4000/emb_6B_50_300_common.npy")
for i in range(2,11):
    model = TSNE(n_components=i, verbose=2, random_state=0)
    print "Reduced to "+str(i)+" dimensions"
    reduced_matrix = model.fit_transform(target_matrix)
    np.save("emb_6B_50_"+str(i)+"_common.npy",reduced_matrix)

# plt.figure(figsize=(200, 200), dpi=100)
# max_x = np.amax(reduced_matrix, axis=0)[0]
# max_y = np.amax(reduced_matrix, axis=0)[1]
# plt.xlim((-max_x,max_x))
# plt.ylim((-max_y,max_y))
#
# plt.scatter(reduced_matrix[:, 0], reduced_matrix[:, 1], 20)
#
# for row_id in range(0, len(rows)):
#     target_word = glove_label[rows[row_id]]
#     x = reduced_matrix[row_id, 0]
#     y = reduced_matrix[row_id, 1]
#     plt.annotate(target_word, (x,y))
#
# plt.show()
print "Done"
# plt.savefig("glove_300d_2000.png")
