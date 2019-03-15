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

data = pd.read_csv(dataPath+'santhosh_peaks_norm_rm32.csv').values
# data = pd.read_csv(dataPath+'santhosh_peaks_dirNorm.csv').values
print('data shape:', data.shape)

# principle component analysis
pca = PCA()
reduced = pca.fit_transform(data)
# mean-shift clustering
clustering = MeanShift().fit(reduced[:, :2])
ms_labels = clustering.labels_

fig1, ax1 = plt.subplots(1)
ax1.scatter(reduced[:, 0], reduced[:, 1], c=ms_labels, s=100,  alpha=.5)
ax1.set_title('Top 2 Principle Components')
ax1.set_xlabel('component 1')
ax1.set_ylabel('component 2')

cumulative = np.cumsum(pca.explained_variance_ratio_)
fig2, ax2 = plt.subplots(1)
ax2.plot(cumulative)
ax2.set_title('Cumulative Information')
ax2.set_xlabel('dimensions')
ax2.set_ylabel('variance explained')

fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
ax3.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2], s=100,  alpha=.5,
            c=ms_labels)
ax3.set_title('Top 3 Principle Components')


# embed with t-SNE into 2 dimensions
X_embed2d = TSNE(
    n_components=2, perplexity=30).fit_transform(reduced[:, :2])
# plot the 2D embedding of the 2D data
fig4, ax4 = plt.subplots(1)
ax4.scatter(X_embed2d[:, 0], X_embed2d[:, 1], c=ms_labels,
            alpha=.5, s=100)
ax4.set_title('TSNE 2D Embedding')
ax4.set_xlabel('dimension 1')
ax4.set_ylabel('dimension 2')

# embed with t-SNE into 3 dimensions
X_embed3d = TSNE(
    n_components=3, perplexity=30).fit_transform(reduced[:, :3])
fig5 = plt.figure()
ax5 = fig5.add_subplot(111, projection='3d')
ax5.scatter(
    X_embed3d[:, 0], X_embed3d[:, 1], X_embed3d[:, 2], c=ms_labels,
    alpha=.5, s=100
)
ax5.set_title('TSNE 3D Embedding')

plt.show()
