import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import MeanShift
import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import linkage, dendrogram

# load data
base = 'D:\\calcium\\'
folder = ''
dataPath = base + folder

filePath = dataPath+'varsha_corr.csv'
data = pd.read_csv(filePath, header=None).values


print('data shape:', data.shape)
dist_array = ssd.squareform(1 - data, checks=False)

# hierarchical clustering
Z = linkage(dist_array, method='ward', optimal_ordering=True)
dendrogram(Z)  # plot clustering as dendrogram
plt.show()

# # principle component analysis
# pca = PCA()
# reduced = pca.fit_transform(data)
# # mean-shift clustering
# # clustering = MeanShift().fit(reduced[:, :2])
# clustering = MeanShift().fit(data)
# ms_labels = clustering.labels_
# print('number of MeanShift clusters:', np.max(ms_labels)+1)

# # Top 2 PCA componenets, plotted with Mean Shift Labels
# plt.scatter(reduced[:, 0], reduced[:, 1], alpha=.5, s=100, c=ms_labels)
# plt.title('with Mean Shift Labels')
# plt.xlabel('component 1')
# plt.ylabel('component 2')
# plt.show()

# # embed with t-SNE into 2 dimensions
# X_embed2d = TSNE(
#     n_components=2, perplexity=30).fit_transform(reduced[:, :])
# # plot the 2D embedding of the 2D data
# fig4, ax4 = plt.subplots(1)
# ax4.scatter(X_embed2d[:, 0], X_embed2d[:, 1], c=ms_labels,
#             alpha=.5, s=100)
# ax4.set_title('TSNE 2D Embedding')
# ax4.set_xlabel('dimension 1')
# ax4.set_ylabel('dimension 2')
# plt.show()