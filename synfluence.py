import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection

# load data
base = 'D:\\NEURONoutput\\'
folder = 'lockedSyn_bigSet3\\'
dataPath = base + folder

condition = ['None', 'E', 'I', 'EI']

VmTreeDF, iCaTreeDF = {}, {}
for c in condition:
    VmTreeDF[c] = pd.read_hdf(dataPath+c+'\\treeRecData.h5', 'Vm')
    iCaTreeDF[c] = pd.read_hdf(dataPath+c+'\\treeRecData.h5', 'iCa')

manipSynsDF = pd.read_csv(dataPath+'manipulatedSyns.csv')
locationsDF = pd.read_csv(dataPath+'treeRecLocations.csv')
synTimesDF = pd.read_hdf(dataPath+'baseSynTimes.h5', 'I')

# data dimensions
# keep in mind that these have likely been downsampled
# ran at 10kHz, downsampled by factor of 10
trials = VmTreeDF['E'].columns.levels[0].values
directions = VmTreeDF['E'].columns.levels[1].values
recInds = VmTreeDF['E'].columns.levels[2].values
tsteps, nrecs = VmTreeDF['E'].shape[0], recInds.shape[0]
numTrials, numDirs = len(trials), len(directions)
dendSegs = int(VmTreeDF['E'].columns.size/numTrials/numDirs/350)

print('num trials:', numTrials, ', directions:', directions)
print('num recs:', nrecs, ', time steps:', tsteps, 'segs per dend:', dendSegs)

synDelay = 10  # synapse activation delay (still 10, see NetCons)


def grabProxRecs(maxDist=50):
    '''
    Indices and cable distances for recs proximal to the manipulated synapses.
    Locations may be used to roughly determine which side of the target synapse
    each recording is on (bi-directional lineplot).
    '''
    distBetwRecs = pd.read_csv(dataPath+'distBetwRecs.csv')
    proxDists, proxLocs = {}, {}
    for dend in manipSynsDF['dendNum']:
        mid = str(int(dend*dendSegs + dendSegs/2))  # calc rec index of syn
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
        for c in condition:
            cols = [(tr, dir, rec) for tr in range(numTrials)
                    for rec in dists[dend].index.values]
            dirRecs[c][dend] = treeRecs[c].loc[:, cols]
    return dirRecs


def timeSlice(dirRecs, dir, lead=3, trail=5):
    '''
    Return slice in time with a given number of timesteps (dur) in leading and
    trailing time around when the locked synapses have their event. This is
    at the time when the bar passes over them, taking offset into account,
    since timing jitter is turned off for these synapses.
    '''
    for s, d in manipSynsDF.values:
        start = int(synTimesDF[dir][s]+synDelay - lead)
        stop = int(synTimesDF[dir][s]+synDelay + trail)
        for c in condition:
            dirRecs[c][d] = dirRecs[c][d].loc[start:stop, :]
    return dirRecs


def binSpace(dirRecs, dists, bin_sz=1, trial='avg'):
    '''
    Bin recordings by distance from their target dendritic site. This way the
    duplicate recordings from each section registered as the same distance are
    eliminated. Still need to look in to why exactly it is an issue in the
    first place.
    '''
    binRecs = {c: {dend: [] for dend in manipSynsDF['dendNum'].values}
               for c in condition}
    binDists = {dend: [] for dend in manipSynsDF['dendNum'].values}
    for d in manipSynsDF['dendNum'].values:
        nBins = int(dists[d].values.max() / bin_sz)
        for j, c in enumerate(condition):
            binRecs[c][d] = []
            # pre-select trial to allow boolean array indexing
            # dirRecs dataframes columns are multi-indexed.
            if trial != 'avg':
                recs = dirRecs[c][d].loc[:, trial]
            else:
                recs = dirRecs[c][d]
            for i in range(nBins):
                # get indices where distance is within the bins nounds
                low = dists[d].values < (i*bin_sz + bin_sz)
                high = dists[d].values >= i*bin_sz
                inds = low*high  # want inds that are true for both
                if inds.any() or 0:
                    binRecs[c][d].append(
                        recs.loc[:, inds].mean(axis=1)
                    )
                    if not j:
                        binDists[d].append(i*bin_sz)
            binRecs[c][d] = pd.DataFrame(np.array(binRecs[c][d]).T,
                                         columns=binDists[d])

    return binRecs


def cableGridPlot(dirRecs, dists, locs,
                  trial='avg', type='surface', plot=False):
    '''
    Grid of plots [surface(3D), waterfall(3D) or heatmap(2D)] for all
    conditions of each locked synapse/dendrite.
    '''
    fig = plt.figure(figsize=(19, 9))
    axes = [
        list(range(len(condition)))
        for _ in range(manipSynsDF['dendNum'].shape[0])
    ]

    dirRecs = averageTrials(dirRecs) if trial == 'avg' else dirRecs

    for i, (syn, dend) in enumerate(manipSynsDF.values):
        ord = np.argsort(dists[dend].values)  # ensure distance is ascending
        timeAx = np.array([dirRecs['E'][dend].index.values
                          for _ in range(dists[dend].shape[0])]).T
        distAx = np.array(
            [dists[dend].values[ord]
                for _ in range(len(dirRecs['E'][dend].index))]
        )
        mid = int(dend*dendSegs + dendSegs/2)  # index for seg
        for j, cond in enumerate(condition):
            if trial != 'avg':
                vals = dirRecs[cond][dend][trial].values[:, ord]
            else:
                vals = dirRecs[cond][dend].values[:, ord]  # sort by distance
            if type == 'surface':
                axes[i][j] = fig.add_subplot(
                                len(condition),  # number cols
                                manipSynsDF['dendNum'].shape[0],  # number rows
                                int(i*len(condition)+j+1),  # position
                                projection='3d'
                            )
                axes[i][j].plot_surface(
                    distAx, timeAx, vals, rstride=1,
                    cstride=1, cmap=plt.cm.coolwarm
                )
            elif type == 'heat':
                axes[i][j] = fig.add_subplot(
                                len(condition),  # number cols
                                manipSynsDF['dendNum'].shape[0],  # number rows
                                int(i*len(condition)+j+1)  # position
                            )
                X, Y = np.meshgrid(dists[dend].values, np.arange(tsteps))
                axes[i][j].pcolormesh(X, Y, vals, cmap=plt.cm.coolwarm)
            elif type == 'water':
                axes[i][j] = fig.add_subplot(
                                len(condition),  # number cols
                                manipSynsDF['dendNum'].shape[0],  # number rows
                                int(i*len(condition)+j+1),  # position
                                projection='3d'
                            )
                axes[i][j] = waterfall(axes[i][j], distAx, timeAx, vals)

            axes[i][j].set_title(
                'syn %d\ndend %d\nXY(%d, %d)'
                % (syn, dend, locs[dend]['X'][mid], locs[dend]['Y'][mid])
            )

    if type != 'water' and 0:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm,
                                   norm=plt.Normalize(
                                    vmin=vals.min(),
                                    vmax=vals.max()))
        sm._A = []
        fig.colorbar(sm)

    fig.tight_layout(pad=0)
    if plot:
        plt.show()

    return fig


def cableLinePlot(dirRecs, dists, locs, trial='avg', plot=True, bin=False,
                  conds=['None', 'E', 'I', 'EI']):
    '''
    1-Dimensional 'line' plots over cable distance that have been collapsed
    over time. Means over the time axis are taken of dataframes that have
    already been sliced around the bar stimulation time by timeSlice().
    '''
    fig1 = plt.figure(figsize=(15, 6))
    axes1 = [[] for _ in range(manipSynsDF['dendNum'].shape[0])]

    # averaging and binning of recordings
    dirRecs = averageTrials(dirRecs) if trial == 'avg' else dirRecs
    dirRecs = binSpace(dirRecs, dists,
                       bin_sz=1, trial=trial) if bin else dirRecs

    cable = {c: {dend: [] for dend in manipSynsDF['dendNum'].values}
             for c in condition}

    for i, (syn, dend) in enumerate(manipSynsDF.values):
        axes1[i] = fig1.add_subplot(
                        1,  # number cols
                        manipSynsDF['dendNum'].shape[0],  # number rows
                        int(i+1),  # position
                        xlabel='distance',
                    )
        mid = int(dend*dendSegs + dendSegs/2)  # index for seg
        for j, cond in enumerate(conds):
            cable[cond][dend] = dirRecs[cond][dend].mean(axis=0)

            if not bin:
                if trial != 'avg':
                    vals = cable[cond][dend][trial].values
                else:
                    vals = cable[cond][dend].values
                dendDists = dists[dend].values
            else:
                vals = cable[cond][dend].values
                dendDists = dirRecs[cond][dend].columns.values
            ord = np.argsort(dendDists)

            axes1[i].plot(dendDists[ord], vals[ord],
                          marker='o', alpha=.5, label=cond)
            axes1[i].set_title(
                'syn %d\ndend %d\nXY(%d, %d)'
                % (syn, dend, locs[dend]['X'][mid], locs[dend]['Y'][mid])
            )

    handles, labels = axes1[0].get_legend_handles_labels()
    fig1.legend(handles, labels, loc='upper center')
    fig1.tight_layout(pad=0)

    if len(conds) == 2:
        fig2 = plt.figure(figsize=(15, 6))
        axes2 = [[] for _ in range(manipSynsDF['dendNum'].shape[0])]

        for i, (syn, dend) in enumerate(manipSynsDF.values):
            axes2[i] = fig2.add_subplot(
                            1,  # number cols
                            manipSynsDF['dendNum'].shape[0],  # number rows
                            int(i+1),  # position
                            xlabel='distance',
                        )
            if not i:
                axes2[i].set_ylabel(
                    'difference (%s - %s)' % (conds[0], conds[1]))
            ord = np.argsort(dists[dend].values)
            mid = int(dend*dendSegs + dendSegs/2)  # index for seg
            if not bin:
                if trial != 'avg':
                    vals1 = cable[conds[0]][dend][trial].values
                    vals2 = cable[conds[1]][dend][trial].values
                else:
                    vals1 = cable[conds[0]][dend].values
                    vals2 = cable[conds[1]][dend].values
                dendDists = dists[dend].values
            else:
                vals1 = cable[conds[0]][dend].values
                vals2 = cable[conds[1]][dend].values
                dendDists = dirRecs[cond][dend].columns.values
            ord = np.argsort(dendDists)
            diff = np.abs(vals1 - vals2)

            axes2[i].plot(dendDists[ord], diff[ord],
                          marker='o', alpha=.5)
            axes2[i].set_title(
                'syn %d\ndend %d\nXY(%d, %d)'
                % (syn, dend, locs[dend]['X'][mid], locs[dend]['Y'][mid])
            )

        fig2.tight_layout(pad=0)
    else:
        fig2 = None

    if plot:
        plt.show()

    return fig1, fig2


def averageTrials(dirRecs):
    '''
    Return an average over trials for all recordings. Takes in dict that is
    already broken into proximal recordings around certain dendrites for each
    condition (only one direction).
    '''
    avgRecs = {c: {dend: [] for dend in manipSynsDF['dendNum'].values}
               for c in condition}
    for dend in manipSynsDF['dendNum'].values:
        for cond in condition:
            avgRecs[cond][dend] = dirRecs[cond][dend].mean(
                                    axis=1, level='synapse'
                                  )

    return avgRecs


def waterfall(ax, X, Y, Z):
    '''
    Simple 3D waterfall plot of timeseries data. Position on X, time going in
    to the plot on Y, value on Z.
    '''
    verts = []
    for i in range(X.shape[1]):
        verts.append(list(zip(Y[:, i], Z[:, i])))

    poly = PolyCollection(verts, facecolors='g', alpha=.3)
    ax.add_collection3d(poly, zs=X[0], zdir='x')
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_zlim(Z.min(), Z.max())

    return ax


def locDists(dists, locs):
    '''
    Visualize the locations of proximal (according to cable distance) recording
    sites relative to each locked synapse/dendrite.
    '''
    fig = plt.figure(figsize=(10, 5))
    axes = [[] for _ in range(manipSynsDF['dendNum'].shape[0])]

    for i, (syn, dend) in enumerate(manipSynsDF.values):
        mid = int(dend*dendSegs + dendSegs/2)  # idx of target syn
        axes[i] = fig.add_subplot(
                        1,  # number cols
                        manipSynsDF['dendNum'].shape[0],  # number rows
                        int(i+1),  # position
                    )
        axes[i].scatter(locs[dend]['X'], locs[dend]['Y'], marker='o', alpha=.5)
        axes[i].scatter(locs[dend]['X'][mid], locs[dend]['Y'][mid],
                        marker='x', alpha=1, s=100)
        axes[i].set_title(
            'syn %d\ndend %d\nXY(%d, %d)'
            % (syn, dend, locs[dend]['X'][mid], locs[dend]['Y'][mid])
        )

    fig.tight_layout(pad=0)
    plt.show()


if __name__ == '__main__':
    proxDists, proxLocs = grabProxRecs(maxDist=50)
    dirVmRecs = getDir(VmTreeDF, proxDists, 0)
    dirVmRecs = timeSlice(dirVmRecs, 0, lead=3, trail=15)

    # surface plot
    # cableFig3D = cableGridPlot(dirVmRecs, proxDists, proxLocs, trial='avg')
    # cableFig3D.savefig(dataPath+'cableGrid_surface.png')

    # heat map
    # cableFig = cableGridPlot(dirVmRecs, proxDists, proxLocs, type='heat')
    # cableFig.savefig(dataPath+'cableGrid_heatmap.png')

    # waterfall
    # cableFig3D = cableGridPlot(dirVmRecs, proxDists, proxLocs,
    #                            trial='avg', type='water')
    # cableFig3D.savefig(dataPath+'cableGrid_waterfall.png')

    # cable line-plot
    cableLine, cableDiff = cableLinePlot(
                            dirVmRecs, proxDists, proxLocs, trial='avg',
                            conds=['E', 'I'], bin=True)
    cableLine.savefig(dataPath+'cableLine.png')
    cableDiff.savefig(dataPath+'cableDiff.png')

    # locDists(proxDists, proxLocs)
