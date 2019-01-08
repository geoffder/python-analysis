import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load data
base = 'C:\\Users\\geoff\\NEURONoutput\\'
folder = 'spike_test\\'
dataPath = base + folder

spikesDF = pd.read_hdf(dataPath+'spikeData.h5', 'dirSpks')
VmDF = pd.read_hdf(dataPath+'spikeData.h5', 'Vm')

trials, directions = spikesDF.columns.values, spikesDF.index.values
numTrials, numDirs = len(trials), len(directions)
print('num trials:', numTrials, 'directions:', directions)

def calcDS(data):
    vals = data.values
    xpts = vals * np.cos(directions*np.pi/180).reshape(numDirs,1)
    ypts = vals * np.sin(directions*np.pi/180).reshape(numDirs,1)
    xsum = xpts.sum(axis=0)
    ysum = ypts.sum(axis=0)

    radius = np.sqrt(xsum**2 + ysum**2)
    DSi = radius / np.sum(vals, axis=0)

    theta = np.arctan2(ysum,xsum) * 180 / np.pi
    # theta[theta < 0] += 360 # maybe leave as centred on 0 (can average)

    return theta, DSi

def polarPlot(trials, directions, data, theta, DSi):
    vals = data.values

    # resorting directions and making circular for polar axes
    inds = np.array(directions).argsort()
    circVals = vals[inds]
    circVals = np.concatenate(
        [circVals, circVals[0,:].reshape(1, len(trials))], axis=0)
    circle = np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315, 0])

    peak = np.max(vals) # to set axis max
    avgTheta, avgDSi = np.deg2rad(np.mean(theta)), np.mean(DSi)
    theta = np.deg2rad(theta) # to radians for plotting

    fig = plt.figure()
    polar = fig.add_subplot(111, projection='polar')
    # plot trials lighter
    polar.plot(circle, circVals, color='.75')
    polar.plot([theta, theta],[np.zeros(len(trials)), DSi*peak], color='.75')
    # plot avg darker
    polar.plot(circle, np.mean(circVals, axis=1), color='.0', linewidth=2)
    polar.plot([avgTheta, avgTheta],[0.0, avgDSi*peak],color='.0', linewidth=2)
    # misc settings
    polar.set_rlabel_position(-22.5) #labels away from line
    polar.set_rmax(peak)
    polar.set_rticks([peak])
    polar.set_thetagrids([0,90,180,270])
    # polar.grid(False)
    # fig.tight_layout()
    plt.show()

    return fig

if __name__ == '__main__':
    theta, DSi = calcDS(spikesDF)
    print('theta:', theta)
    print('DSi:', DSi)
    fig = polarPlot(trials, directions, spikesDF, theta, DSi)
    fig.savefig(dataPath+'polar.svg')
