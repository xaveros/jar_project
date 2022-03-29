import matplotlib.pyplot as plt
import numpy as np
import pylab
from IPython import embed
from scipy.optimize import curve_fit
from matplotlib.mlab import specgram
import os

from jar_functions import import_data
from jar_functions import import_amfreq

from scipy.optimize import curve_fit
from jar_functions import sin_response
from jar_functions import mean_noise_cut
from jar_functions import gain_curve_fit

plt.rcParams.update({'font.size': 12})

def take_second(elem):      # function for taking the names out of files
    return elem[1]

identifier = ['2018lepto5']
for ident in identifier:

    predict = []

    rootmeansquare = []
    threshold = []

    gain = []
    mgain = []
    phaseshift = []
    mphaseshift = []
    amfreq = []
    amf = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]

    currf = None
    idxlist = []

    data = sorted(np.load('%s files.npy' %ident), key = take_second)      # list with filenames in it

    for i, d in enumerate(data):
        dd = list(d)

        jar = np.load('%s.npy' %dd)     # load data for every file name
        jm = jar - np.mean(jar)         # low-pass filtering by subtracting mean
        print(dd)

        time = np.load('%s time.npy' %dd)       # time file
        dt = time[1] - time[0]

        n = int(1/float(d[1])/dt)
        cutf = mean_noise_cut(jm, n = n)
        cutt = time

        sinv, sinc = curve_fit(sin_response, time, jm - cutf, [float(d[1]), 2, 0.5])        # fitting
        print('frequency, phaseshift, amplitude:', sinv)
        p = sinv[1]
        A = np.sqrt(sinv[2] ** 2)
        f = float(d[1])
        if sinv[2] < 0:
            p = p + np.pi
        phaseshift.append(p)
        gain.append(A)
        if f not in amfreq:
            amfreq.append(f)

        # root mean square
        RMS = np.sqrt(np.mean(((jm - cutf) - sin_response(cutt, sinv[0], sinv[1], sinv[2]))**2))
        thresh = A / np.sqrt(2)

        # mean over same amfreqs for phase and gain
        if currf is None or currf == d[1]:
            currf = d[1]
            idxlist.append(i)

        else:  # currf != f
            meanf = []  # lists to make mean of
            meanp = []
            meanrms = []
            meanthresh = []
            for x in idxlist:
                meanf.append(gain[x])
                meanp.append(phaseshift[x])
                meanrms.append(RMS)
                meanthresh.append(thresh)
            meanedf = np.mean(meanf)
            meanedp = np.mean(meanp)
            meanedrms = np.mean(meanrms)
            meanedthresh = np.mean(meanthresh)

            mgain.append(meanedf)
            mphaseshift.append(meanedp)
            rootmeansquare.append(meanedrms)
            threshold.append(meanedthresh)
            currf = d[1]    # set back for next loop
            idxlist = [i]
    meanf = []
    meanp = []
    meanrms = []
    meanthresh = []
    for y in idxlist:
        meanf.append(gain[y])
        meanp.append(phaseshift[y])
        meanrms.append(RMS)
        meanthresh.append(thresh)
    meanedf = np.mean(meanf)
    meanedp = np.mean(meanp)
    meanedrms = np.mean(meanrms)
    meanedthresh = np.mean(meanthresh)

    mgain.append(meanedf)
    mphaseshift.append(meanedp)
    rootmeansquare.append(meanedrms)
    threshold.append(meanedthresh)

    # as arrays
    mgain_arr = np.array(mgain)
    mphaseshift_arr = np.array(mphaseshift)
    amfreq_arr = np.array(amfreq)
    rootmeansquare_arr = np.array(rootmeansquare)
    threshold_arr = np.array(threshold)

    # condition needed to be fulfilled: RMS < threshold or RMS < mean(RMS)
    idx_arr = (rootmeansquare_arr < threshold_arr) | (rootmeansquare_arr < np.mean(rootmeansquare_arr))

    fig = plt.figure(figsize = (8.27, 11.69))
    fig.suptitle('gaincurve and RMS %s' %ident)
    ax0 = fig.add_subplot(2, 1, 1)
    ax0.plot(amfreq_arr, mgain_arr, 'o')
    ax0.set_yscale('log')
    ax0.set_xscale('log')
    ax0.set_ylabel('gain [Hz/(mV/cm)]')
    ax0.axes.xaxis.set_ticklabels([])
    plt.text(-0.1, 1.05, "A)", fontweight=550, transform=ax0.transAxes)

    ax1 = fig.add_subplot(2, 1, 2, sharex = ax0)
    ax1.plot(amfreq, threshold, 'o-', label = 'threshold', color = 'b')
    ax1.set_xscale('log')
    ax1.plot(amfreq, rootmeansquare, 'o-', label = 'RMS', color ='orange')
    ax1.set_xscale('log')
    ax1.set_xlabel('envelope frequency [Hz]')
    ax1.set_ylabel('RMS [Hz]')
    plt.text(-0.1, 1.05, "B)", fontweight=550, transform=ax1.transAxes)
    plt.legend()
    pylab.show()
    #fig.savefig('test.pdf')
    #np.save('phaseshift_%s' % ident, mphaseshift_arr[idx_arr])
    #np.save('gain_%s' %ident, mgain_arr[idx_arr])
    #np.save('amf_%s' %ident, amfreq_arr[idx_arr])

embed()