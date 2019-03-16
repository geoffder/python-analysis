import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import MeanShift

# load data
base = 'D:\\calcium\\'
folder = ''
dataPath = base + folder
# filePath = dataPath+'santhosh_peaks_dir.csv'
# filePath = dataPath+'santhosh_peaks_dir_rm33.csv'
# filePath = dataPath+'santhosh_peaks_norm.csv'
filePath = dataPath+'santhosh_peaks_norm_rm33.csv'
data = pd.read_csv(filePath, header=None).values

print('data shape:', data.shape)

# principle component analysis
pca = PCA()
reduced = pca.fit_transform(data)
# mean-shift clustering
clustering = MeanShift().fit(reduced[:, :2])
ms_labels = clustering.labels_

# Top 2 PCA componenets, plotted with ROI number
fig1, ax1 = plt.subplots(1, 3, figsize=(14, 6))
for i in range(reduced.shape[0]):
    ax1[0].scatter(reduced[i, 0], reduced[i, 1], alpha=.5, s=100, c='0',
                   marker='$%s$' % i)
ax1[0].set_title('with ROI numbers')
ax1[0].set_xlabel('component 1')
ax1[0].set_ylabel('component 2')
# Top 2 PCA componenets, plotted with Mean Shift Labels
ax1[1].scatter(reduced[:, 0], reduced[:, 1], alpha=.5, s=100, c=ms_labels)
ax1[1].set_title('with Mean Shift Labels')
ax1[1].set_xlabel('component 1')
ax1[1].set_ylabel('component 2')
# cumulative variance explained
cumulative = np.cumsum(pca.explained_variance_ratio_)
ax1[2].plot(cumulative)
ax1[2].set_title('Cumulative Information')
ax1[2].set_xlabel('dimensions')
ax1[2].set_ylabel('variance explained')
fig1.tight_layout()

# Top 3 PCA componenets, plotted with Mean Shift Labels
fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
ax3.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2], alpha=.5, s=100,
            c=ms_labels)
ax3.set_title('Top 3 Principle Components')


# embed with t-SNE into 2 dimensions
X_embed2d = TSNE(
    n_components=2, perplexity=30).fit_transform(reduced[:, :])
# plot the 2D embedding of the 2D data
fig4, ax4 = plt.subplots(1)
ax4.scatter(X_embed2d[:, 0], X_embed2d[:, 1], c=ms_labels,
            alpha=.5, s=100)
ax4.set_title('TSNE 2D Embedding')
ax4.set_xlabel('dimension 1')
ax4.set_ylabel('dimension 2')

# embed with t-SNE into 3 dimensions
X_embed3d = TSNE(
    n_components=3, perplexity=30).fit_transform(reduced[:, :])
fig5 = plt.figure()
ax5 = fig5.add_subplot(111, projection='3d')
ax5.scatter(
    X_embed3d[:, 0], X_embed3d[:, 1], X_embed3d[:, 2], c=ms_labels,
    alpha=.5, s=100
)
ax5.set_title('TSNE 3D Embedding')

plt.show()
