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

identifier = ['2018lepto98']
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
        if dd[1] == '0.05':
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
            A = sinv[2]
            if A < 0:
                p = p + np.pi
                A = -A
            f = float(d[1])
            phaseshift.append(p)
            gain.append(A)
            if f not in amfreq:
                amfreq.append(f)

            # jar trace
            plt.plot(time, jar, color = 'C0')
            #plt.hlines(y=np.min(jar) - 2, xmin=0, xmax=400, lw=2.5, color='r', label='stimulus duration')
            plt.title('JAR trace 2018lepto98, AM-frequency: %sHz' % float(d[1]))
            plt.xlabel('time[s]')
            plt.ylabel('frequency[Hz]')
            plt.show()

            # low pass filter by mean subtraction
            # plt.plot(time, jm)
            # plt.title('JAR trace: filtered by mean subtraction')
            # plt.xlabel('time[s]')
            # plt.ylabel('frequency[Hz]')
            # plt.show()

            # filter by running average
            fig = plt.figure(figsize = (8.27,11.69))
            fig.suptitle('JAR trace spectogram 2018lepto98:\n subtraction of mean and running average')
            ax = fig.add_subplot(211)
            ax.plot(time, jm, color = 'C0', label = '1)')
            ax.plot(time, jm - cutf, color = 'darkorange', label = '2)')
            ax.set_ylabel('frequency[Hz]')
            ax.set_ylim(-10.5, 10.5)
            ax.axes.xaxis.set_ticklabels([])
            plt.legend(loc='upper right')
            plt.text(-0.1, 1.05, "A)", fontweight=550, transform=ax.transAxes)

            # jar trace and fit
            ax1 = fig.add_subplot(212)
            ax1.plot(time, jm - cutf, color = 'darkorange', label = '2)')
            phase_gain = [(((p % (2 * np.pi)) * 360) / (2 * np.pi)), A]
            print(phase_gain)
            ax1.plot(time, sin_response(time, *sinv), color = 'forestgreen', label='3)')

            ax1.set_xlabel('time[s]')
            ax1.set_ylabel('frequency[Hz]')
            ax1.set_ylim(-10.5,10.5)
            plt.legend(loc = 'upper right')
            plt.text(-0.1, 1.05, "B)", fontweight=550, transform=ax1.transAxes)
            plt.show()
            plt.savefig('test_fig.png')
            embed()
