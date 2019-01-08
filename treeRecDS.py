import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load data
base = 'C:\\Users\\geoff\\NEURONoutput\\'
folder = 'tree_test\\'
dataPath = base + folder

VmTreeDF = pd.read_hdf(dataPath+'treeRecData.h5', 'Vm')
iCaTreeDF = pd.read_hdf(dataPath+'treeRecData.h5', 'iCa')
locationsDF = pd.read_csv(dataPath+'treeRecLocations.csv')
distBetwRecsDF = pd.read_csv(dataPath+'distBetwRecs.csv')

# data dimensions
trials, directions = VmTreeDF.columns.levels[0].values, VmTreeDF.columns.levels[1].values
tsteps, nrecs = VmTreeDF.shape[0], locationsDF.shape[0]
numTrials, numDirs = len(trials), len(directions)
print('num trials:', numTrials, ', directions:', directions)
print('num recs:', nrecs, ', time steps:', tsteps)

# global params
threshVm = -50

def calcTreeDS(data, thresh):
    vals = data.values
    vals[vals < threshVm] = thresh

    area = np.sum((vals - thresh), axis=0).reshape(nrecs, numDirs, numTrials)
    print(area.shape)

    xpts = area * np.cos(directions*np.pi/180).reshape(numDirs,1)
    ypts = area * np.sin(directions*np.pi/180).reshape(numDirs,1)
    xsum = xpts.sum(axis=1)
    ysum = ypts.sum(axis=1)

    radius = np.sqrt(xsum**2 + ysum**2)
    DSi = radius / np.sum(area, axis=1)
    theta = np.arctan2(ysum,xsum) * 180 / np.pi

    return theta, DSi

def plotMap(locs, theta, DSi, degMax=180, dsiMax=1, trial=-1):
    locs = locs.values

    # calculate avgs and concatenate as new columns
    theta = np.concatenate([theta, theta.mean(axis=1,keepdims=1)], axis=1)
    theta = np.abs(theta) # linearize, diff from preffered
    DSi = np.concatenate([DSi, DSi.mean(axis=1,keepdims=1)], axis=1)

    # apply thresholds (colour range)
    theta[theta > degMax] = degMax
    DSi[DSi > dsiMax] = dsiMax
    # normalize data
    theta = theta / degMax
    DSi = DSi / dsiMax

    tFig, thetaMap = plt.subplots(1)
    thetaMap.scatter(
        locs[:,0],locs[:,1], c=theta[:,trial],
        cmap='jet', alpha=.5)
    thetaMap.set_title('Theta (vs Pref)')
    dFig, DSiMap = plt.subplots(1)
    DSiMap.scatter(locs[:,0], locs[:,1], c=DSi[:,trial],
        cmap='jet', alpha=.5)
    DSiMap.set_title('DSi')

    # create theta colour bar (legend)
    sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=degMax))
    sm._A = []
    tFig.colorbar(sm)
    # create DSi colour bar (legend)
    sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=dsiMax))
    sm._A = []
    dFig.colorbar(sm)

    # fig.tight_layout()
    plt.show()

    return tFig, dFig

# create function for correlating activity between recs over the tree
# create function for "line plots" of voltage along a rolled out section of dendrite

if __name__ == '__main__':
    theta, DSi = calcTreeDS(VmTreeDF, threshVm)
    # print('theta:', theta)
    # print('DSi:', DSi)
    thetaFig, DSiFig = plotMap(locationsDF, theta, DSi, degMax=180)

# multiindex note
# this will pull all the whole duration (:) trials and all dirs for rec 0
# data.loc[:, (slice(None), slice(None), 0)]
