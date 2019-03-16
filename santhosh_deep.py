import pandas as pd

import matplotlib.pyplot as plt
from cycler import cycler
from mpl_toolkits.mplot3d import Axes3D

from deep_ae_class_pt_mod import DeepAutoEncoder
from sklearn.manifold import TSNE
from sklearn.cluster import MeanShift


def main(load=False):
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

    # hidden_layer_sizes = [40, 16, 2]
    hidden_layer_sizes = [60, 16, 2]  # best?
    # hidden_layer_sizes = [16, 2]
    # hidden_layer_sizes = [60, 16, 3]
    # hidden_layer_sizes = [16, 3]

    DAE = DeepAutoEncoder(hidden_layer_sizes, activation='tanh')
    DAE.fit(data, lr=1e-3, epochs=1000, batch_sz=30)

    reduced = DAE.get_reduced(data)
    print('reduced data shape:', reduced.shape)

    clustering = MeanShift().fit(reduced)
    ms_labels = clustering.labels_
    print('mean-shift labels:', ms_labels)

    if hidden_layer_sizes[-1] == 2:
        fig, ax = plt.subplots(1, 2)
        for i in range(len(reduced)):
            ax[0].scatter(reduced[i, 0], reduced[i, 1], alpha=.5, s=100,
                          marker='$%s$' % i, c='r')
        ax[1].scatter(reduced[:, 0], reduced[:, 1], alpha=.5, s=100,
                      c=ms_labels)
        ax[0].set_title('Reduced 2D, ROI labels')
        ax[0].set_xlabel('component 1')
        ax[0].set_ylabel('component 2')
        ax[1].set_title('Reduced 2D, MeanShift labels')
        ax[1].set_xlabel('component 1')
        ax[1].set_ylabel('component 2')
        fig.tight_layout()
        plt.show()
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2], alpha=.5, s=80)
        plt.show()

    if 1:
        # embed with t-SNE into 2 dimensions
        X_embed2d = TSNE(
            n_components=2, perplexity=30).fit_transform(reduced)
        # plot the 2D embedding of the 2D data
        fig4, ax4 = plt.subplots(1)
        ax4.scatter(X_embed2d[:, 0], X_embed2d[:, 1],
                    alpha=.5, s=100, c=ms_labels)
        ax4.set_title('TSNE 2D Embedding')
        ax4.set_xlabel('dimension 1')
        ax4.set_ylabel('dimension 2')
        plt.show()
    if 0:
        # embed with t-SNE into 3 dimensions
        X_embed3d = TSNE(
            n_components=3, perplexity=30).fit_transform(reduced)
        fig5 = plt.figure()
        ax5 = fig5.add_subplot(111, projection='3d')
        ax5.scatter(
            X_embed3d[:, 0], X_embed3d[:, 1], X_embed3d[:, 2],
            alpha=.5, s=100, c=ms_labels
        )
        ax5.set_title('TSNE 3D Embedding')
        plt.show()


if __name__ == '__main__':
    # change colours used for sequential plotting
    new_colors = [plt.get_cmap('jet')(1. * i/10) for i in range(10)]
    plt.rc('axes', prop_cycle=(cycler('color', new_colors)))

    main()
