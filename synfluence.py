import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection

# load data
base = 'D:\\NEURONoutput\\'
folder = 'lockedSyn\\'
dataPath = base + folder

condition = ['None', 'E', 'I', 'EI']

VmTreeDF, iCaTreeDF = [], []
for c in condition:
    VmTreeDF.append(pd.read_hdf(dataPath+c+'\\treeRecData.h5', 'Vm'))
    iCaTreeDF.append(pd.read_hdf(dataPath+c+'\\treeRecData.h5', 'iCa'))

manipSynsDF = pd.read_csv(dataPath+'manipulatedSyns.csv')
locationsDF = pd.read_csv(dataPath+'treeRecLocations.csv')

# data dimensions
# keep in mind that these have likely been downsampled
# ran at 10kHz, downsampled by factor of 10
trials = VmTreeDF[0].columns.levels[0].values
directions = VmTreeDF[0].columns.levels[1].values
recInds = VmTreeDF[0].columns.levels[2].values
tsteps, nrecs = VmTreeDF[0].shape[0], recInds.shape[0]
numTrials, numDirs = len(trials), len(directions)
print('num trials:', numTrials, ', directions:', directions)
print('num recs:', nrecs, ', time steps:', tsteps)
dendSegs = int(VmTreeDF[0].columns.size/numTrials/numDirs/350)


def grabProxRecs(maxDist=50):
    '''
    Indices and cable distances for recs proximal to the manipulated synapses.
    Locations may be used to roughly determine which side of the target synapse
    each recording is on (bi-directional lineplot).
    '''
    distBetwRecs = pd.read_csv(dataPath+'distBetwRecs.csv')
    proxDists, proxLocs = {}, {}
    for dend in manipSynsDF['dendNum']:
        mid = str(int(dend*dendSegs + dendSegs/2))
        proxDists[dend] = distBetwRecs[mid][distBetwRecs[mid] < maxDist]
        proxLocs[dend] = locationsDF.loc[proxDists[dend].index.values]

    return proxDists, proxLocs


def getDir(treeRecs, dists, dir):
    '''
    Take dataframe of tree recordings and return a new dataframe with only the
    specified direction and recordings proximal to the manipulated synapses.
    '''
    dirRecs = {c: {} for c in condition}
    for dend in manipSynsDF['dendNum']:
        for i, c in enumerate(condition):
            dirRecs[c][dend] = treeRecs[i].drop(
                columns=set(directions).difference([dir]),
                level='direction'
            ).drop(
                columns=set(recInds).difference(dists[dend].index),
                level='synapse'
            )

    return dirRecs


def cableGridPlot(dirRecs, dists, locs,
                  trial='avg', type='surface', plot=False):

    fig = plt.figure(figsize=(19, 9))
    axes = [
        list(range(len(condition)))
        for _ in range(manipSynsDF['dendNum'].shape[0])
    ]

    dirRecs = averageTrials(dirRecs) if trial == 'avg' else dirRecs

    for i, dend in enumerate(manipSynsDF['dendNum'].values):
        timeAx = np.array(
                    [np.arange(tsteps) for _ in range(dists[dend].shape[0])]).T
        distAx = np.array([dists[dend].values for _ in range(tsteps)])
        for j, cond in enumerate(condition):
            if trial != 'avg':
                vals = dirRecs[cond][dend][trial].values
            else:
                vals = dirRecs[cond][dend].values

            if type == 'surface':
                axes[i][j] = fig.add_subplot(
                                len(condition),  # number rows
                                manipSynsDF['dendNum'].shape[0],  # number cols
                                int(i*len(condition)+j+1),  # position
                                projection='3d'
                            )
                axes[i][j].plot_surface(distAx, timeAx, vals, rstride=1,
                                        cstride=1, cmap=plt.cm.coolwarm)
            elif type == 'heat':
                axes[i][j] = fig.add_subplot(
                                len(condition),  # number rows
                                manipSynsDF['dendNum'].shape[0],  # number cols
                                int(i*len(condition)+j+1)  # position
                            )
                X, Y = np.meshgrid(dists[dend].values, np.arange(tsteps))
                axes[i][j].pcolormesh(X, Y, vals, cmap=plt.cm.coolwarm)
            elif type == 'water':
                axes[i][j] = fig.add_subplot(
                                len(condition),  # number rows
                                manipSynsDF['dendNum'].shape[0],  # number cols
                                int(i*len(condition)+j+1),  # position
                                projection='3d'
                            )
                axes[i][j] = waterfall(axes[i][j], distAx, timeAx, vals)

            axes[i][j].set_title(cond + ', dend: ' + str(dend))

    if type != 'water' and 0:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm,
                                   norm=plt.Normalize(
                                    vmin=vals.min(),
                                    vmax=vals.max()))
        sm._A = []
        fig.colorbar(sm)

    fig.tight_layout()
    if plot:
        plt.show()

    return fig


def averageTrials(dirRecs):
    avgRecs = {c: {dend: [] for dend in manipSynsDF['dendNum'].values}
               for c in condition}
    for dend in manipSynsDF['dendNum'].values:
        for cond in condition:
            avgRecs[cond][dend] = dirRecs[cond][dend].mean(
                                    axis=1, level='synapse'
                                  )

    return avgRecs


def waterfall(ax, X, Y, Z):

    verts = []
    for i in range(X.shape[1]):
        verts.append(list(zip(Y[:, i], Z[:, i])))

    poly = PolyCollection(verts, facecolors='g', alpha=.3)
    ax.add_collection3d(poly, zs=X[0], zdir='x')
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_zlim(Z.min(), Z.max())

    return ax


if __name__ == '__main__':
    proxDists, proxLocs = grabProxRecs(maxDist=5)
    dirVmRecs = getDir(VmTreeDF, proxDists, 0)

    cableFig3D = cableGridPlot(dirVmRecs, proxDists, proxLocs, trial='avg')
    cableFig3D.savefig(dataPath+'cableGrid_surface.png')

    # cableFig = cableGridPlot(dirVmRecs, proxDists, proxLocs, type='heat')
    # cableFig.savefig(dataPath+'cableGrid_heatmap.png')

    cableFig3D = cableGridPlot(dirVmRecs, proxDists, proxLocs,
                               trial='avg', type='water')
    cableFig3D.savefig(dataPath+'cableGrid_waterfall.png')
