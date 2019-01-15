import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load data
base = 'C:\\Users\\geoff\\NEURONoutput\\'
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


def linePlots(proxDists, proxLocs, dir):
    lineVmRecs = {c: {} for c in condition}
    for dend in manipSynsDF['dendNum']:
        for i, c in enumerate(condition):
            lineVmRecs[c][dend] = VmTreeDF[i].drop(
                columns=set(directions).difference([dir]),
                level='direction'
            ).drop(
                columns=set(recInds).difference(proxDists[dend].index),
                level='synapse'
            )

    return lineVmRecs


if __name__ == '__main__':
    proxDists, proxLocs = grabProxRecs()
    lineVmRecs = linePlots(proxDists, proxLocs, 180)
